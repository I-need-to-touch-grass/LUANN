-- network.lua
-- Feed-Forward Neural Network (v2: Adam Optimizer, Dropout, Leaky ReLU, Grad Clipping)

local Matrix = require("matrix")
local Network = {}
Network.__index = Network

function Network.new(layer_sizes, config)
	local self = setmetatable({}, Network)
	self.layer_sizes = layer_sizes
	self.num_layers = #layer_sizes - 1

	-- Hyperparameters
	self.lr = (config and config.lr) or 0.005
	self.beta1 = (config and config.beta1) or 0.9
	self.beta2 = (config and config.beta2) or 0.999
	self.eps = 1e-8
	self.l2_reg = (config and config.l2_reg) or 0.0005
	self.dropout = (config and config.dropout) or 0.15
	self.clip_norm = (config and config.clip_norm) or 5.0
	self.leaky_alpha = (config and config.leaky_alpha) or 0.01

	self.t = 0 -- Adam timestep counter
	self.weights = {}
	self.biases = {}

	-- Adam first moment (m) and second moment (v) buffers
	self.m_w, self.v_w = {}, {}
	self.m_b, self.v_b = {}, {}

	for i = 1, self.num_layers do
		local in_dim = layer_sizes[i]
		local out_dim = layer_sizes[i + 1]
		self.weights[i] = Matrix.random(in_dim, out_dim)
		self.biases[i] = Matrix.new(1, out_dim, 0.01)

		self.m_w[i] = Matrix.new(in_dim, out_dim, 0.0)
		self.v_w[i] = Matrix.new(in_dim, out_dim, 0.0)
		self.m_b[i] = Matrix.new(1, out_dim, 0.0)
		self.v_b[i] = Matrix.new(1, out_dim, 0.0)
	end

	return self
end

-- Forward pass with optional Inverted Dropout during training
function Network:forward(X, is_training)
	self.cache_a = { X }
	self.cache_z = {}
	self.drop_masks = {}

	local current = X
	for i = 1, self.num_layers do
		local z = current:dot(self.weights[i]):add(self.biases[i])
		self.cache_z[i] = z
		if i < self.num_layers then
			current = z:leaky_relu(self.leaky_alpha)
			if is_training and self.dropout > 0 then
				local mask = Matrix.dropout_mask(current.rows, current.cols, self.dropout)
				self.drop_masks[i] = mask
				current = current:mul(mask)
			end
		else
			current = z:softmax()
		end
		self.cache_a[i] = current
	end

	return current
end

function Network:compute_loss(probs, Y_true)
	local batch_size = probs.rows
	local eps = 1e-12
	local data_loss = 0.0

	for r = 1, batch_size do
		for c = 1, probs.cols do
			if Y_true.data[r][c] > 0 then
				local p = math.max(probs.data[r][c], eps)
				data_loss = data_loss - Y_true.data[r][c] * math.log(p)
			end
		end
	end
	data_loss = data_loss / batch_size

	local reg_loss = 0.0
	if self.l2_reg > 0 then
		for i = 1, self.num_layers do
			local w = self.weights[i]
			for r = 1, w.rows do
				for c = 1, w.cols do
					reg_loss = reg_loss + w.data[r][c] ^ 2
				end
			end
		end
		reg_loss = 0.5 * self.l2_reg * reg_loss
	end

	return data_loss + reg_loss
end

-- Backward pass with Adam Optimizer, Gradient Clipping & Dropout Backprop
function Network:backward(Y_true)
	local batch_size = Y_true.rows
	local inv_batch = 1.0 / batch_size
	self.t = self.t + 1

	local bias_corr1 = 1.0 - (self.beta1 ^ self.t)
	local bias_corr2 = 1.0 - (self.beta2 ^ self.t)

	local delta = self.cache_a[self.num_layers]:sub(Y_true):scale(inv_batch)

	for i = self.num_layers, 1, -1 do
		local a_prev = self.cache_a[i - 1]
		local grad_w = a_prev:transpose():dot(delta)
		local grad_b = delta:sum_rows()

		if self.l2_reg > 0 then
			grad_w = grad_w:add(self.weights[i]:scale(self.l2_reg))
		end

		-- Clip gradients to prevent spikes
		grad_w = grad_w:clip(-self.clip_norm, self.clip_norm)
		grad_b = grad_b:clip(-self.clip_norm, self.clip_norm)

		-- Propagate error to previous hidden layer
		if i > 1 then
			local d_hidden = delta:dot(self.weights[i]:transpose())
			if self.drop_masks[i - 1] then
				d_hidden = d_hidden:mul(self.drop_masks[i - 1])
			end
			local d_act = self.cache_z[i - 1]:leaky_relu_deriv(self.leaky_alpha)
			delta = d_hidden:mul(d_act)
		end

		-- Adam moment updates for weights
		self.m_w[i] = self.m_w[i]:scale(self.beta1):add(grad_w:scale(1.0 - self.beta1))
		self.v_w[i] = self.v_w[i]:scale(self.beta2):add(grad_w:mul(grad_w):scale(1.0 - self.beta2))

		local m_w_hat = self.m_w[i]:scale(1.0 / bias_corr1)
		local v_w_hat = self.v_w[i]:scale(1.0 / bias_corr2)
		local step_w = m_w_hat:div(v_w_hat:sqrt():add_scalar(self.eps)):scale(self.lr)
		self.weights[i] = self.weights[i]:sub(step_w)

		-- Adam moment updates for biases
		self.m_b[i] = self.m_b[i]:scale(self.beta1):add(grad_b:scale(1.0 - self.beta1))
		self.v_b[i] = self.v_b[i]:scale(self.beta2):add(grad_b:mul(grad_b):scale(1.0 - self.beta2))

		local m_b_hat = self.m_b[i]:scale(1.0 / bias_corr1)
		local v_b_hat = self.v_b[i]:scale(1.0 / bias_corr2)
		local step_b = m_b_hat:div(v_b_hat:sqrt():add_scalar(self.eps)):scale(self.lr)
		self.biases[i] = self.biases[i]:sub(step_b)
	end
end

function Network:predict(X)
	local probs = self:forward(X, false)
	local indices = probs:argmax_rows()
	local preds = {}
	local confidences = {}

	for r = 1, #indices do
		local idx = indices[r]
		preds[r] = idx - 1
		confidences[r] = probs.data[r][idx]
	end

	return preds, confidences, probs
end

function Network:evaluate(X, Y_true, labels)
	local probs = self:forward(X, false)
	local loss = self:compute_loss(probs, Y_true)
	local indices = probs:argmax_rows()
	local correct = 0

	for i = 1, #labels do
		if (indices[i] - 1) == labels[i] then
			correct = correct + 1
		end
	end

	return correct / #labels, loss
end

return Network
