-- network.lua
-- Feed-Forward Neural Network with Backpropagation and SGD + Momentum
-- I mean really all I did was took a python file and just started to write it but in lua using lua functions methods and all the differences between it an python its just like a big refernce from the python file
-- Note: I only created python reference files for matrix.lua and network.lua because they were the ones I was struggling with the most

local Matrix = require("matrix")
local Network = {}
Network.__index = Network

-- Create a new MLP given layer sizes, e.g., {64, 48, 24, 10}
function Network.new(layer_sizes, config)
	local self = setmetatable({}, Network)
	self.layer_sizes = layer_sizes
	self.num_layers = #layer_sizes - 1
	self.lr = (config and config.lr) or 0.05
	self.momentum = (config and config.momentum) or 0.9
	self.l2_reg = (config and config.l2_reg) or 0.0005

	self.weights = {}
	self.biases = {}
	self.vel_w = {}
	self.vel_b = {}

	for i = 1, self.num_layers do
		local in_dim = layer_sizes[i]
		local out_dim = layer_sizes[i + 1]
		self.weights[i] = Matrix.random(in_dim, out_dim)
		self.biases[i] = Matrix.new(1, out_dim, 0.01)
		self.vel_w[i] = Matrix.new(in_dim, out_dim, 0.0)
		self.vel_b[i] = Matrix.new(1, out_dim, 0.0)
	end

	return self
end

-- Forward pass through all layers; caches activations for backprop
function Network:forward(X)
	self.cache_a = { [0] = X }
	self.cache_z = {}

	local current = X
	for i = 1, self.num_layers do
		local z = current:dot(self.weights[i]):add(self.biases[i])
		self.cache_z[i] = z
		if i < self.num_layers then
			current = z:relu()
		else
			current = z:softmax()
		end
		self.cache_a[i] = current
	end

	return current
end

-- Compute Cross-Entropy Loss + L2 Regularization Penalty
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

	-- L2 regularization loss
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

-- Backward pass & SGD with Momentum parameter update
function Network:backward(Y_true)
	local batch_size = Y_true.rows
	local inv_batch = 1.0 / batch_size

	-- Gradient of Cross-Entropy + Softmax simplifies to (P - Y) / N
	local delta = self.cache_a[self.num_layers]:sub(Y_true):scale(inv_batch)

	for i = self.num_layers, 1, -1 do
		local a_prev = self.cache_a[i - 1]
		local grad_w = a_prev:transpose():dot(delta)
		local grad_b = delta:sum_rows()

		-- Add L2 weight decay gradient
		if self.l2_reg > 0 then
			grad_w = grad_w:add(self.weights[i]:scale(self.l2_reg))
		end

		-- Propagate error to previous hidden layer before updating weights
		if i > 1 then
			local d_hidden = delta:dot(self.weights[i]:transpose())
			local d_relu = self.cache_z[i - 1]:relu_deriv()
			delta = d_hidden:mul(d_relu)
		end

		-- Update velocity & apply weights
		self.vel_w[i] = self.vel_w[i]:scale(self.momentum):sub(grad_w:scale(self.lr))
		self.vel_b[i] = self.vel_b[i]:scale(self.momentum):sub(grad_b:scale(self.lr))

		self.weights[i] = self.weights[i]:add(self.vel_w[i])
		self.biases[i] = self.biases[i]:add(self.vel_b[i])
	end
end

-- Predict digit class (0-9) and confidence scores for input matrix X
function Network:predict(X)
	local probs = self:forward(X)
	local indices = probs:argmax_rows()
	local preds = {}
	local confidences = {}

	for r = 1, #indices do
		local idx = indices[r]
		preds[r] = idx - 1 -- Convert 1-based index to 0-9 digit
		confidences[r] = probs.data[r][idx]
	end

	return preds, confidences, probs
end

-- Evaluate accuracy and average loss on a full dataset matrix pair
function Network:evaluate(X, Y_true, labels)
	local probs = self:forward(X)
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
