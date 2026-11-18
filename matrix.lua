-- matrix.lua
-- Pure Lua 2D Matrix & Math Library (v2: Leaky ReLU, Dropout, Element-wise Ops)

local Matrix = {}
Matrix.__index = Matrix

math.randomseed(os.time())

function Matrix.new(rows, cols, init_val)
	local self = setmetatable({}, Matrix)
	self.rows = rows
	self.cols = cols
	self.data = {}
	local val = init_val or 0.0
	for r = 1, rows do
		self.data[r] = {}
		for c = 1, cols do
			self.data[r][c] = val
		end
	end
	return self
end

function Matrix.from_table(tbl)
	local rows = #tbl
	local cols = #tbl
	local m = Matrix.new(rows, cols)
	for r = 1, rows do
		for c = 1, cols do
			m.data[r][c] = tbl[r][c]
		end
	end
	return m
end

-- He uniform initialization tuned for Leaky ReLU
function Matrix.random(rows, cols, scale)
	local m = Matrix.new(rows, cols)
	local s = scale or math.sqrt(2.0 / rows)
	for r = 1, rows do
		for c = 1, cols do
			m.data[r][c] = (math.random() * 2.0 - 1.0) * s
		end
	end
	return m
end

-- Generate an Inverted Dropout mask (scales kept neurons by 1 / (1 - drop_rate))
function Matrix.dropout_mask(rows, cols, drop_rate)
	local m = Matrix.new(rows, cols)
	local keep_prob = 1.0 - drop_rate
	local scale = 1.0 / keep_prob
	for r = 1, rows do
		for c = 1, cols do
			m.data[r][c] = (math.random() < keep_prob) and scale or 0.0
		end
	end
	return m
end

function Matrix:copy()
	local m = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			m.data[r][c] = self.data[r][c]
		end
	end
	return m
end

function Matrix:dot(other)
	assert(
		self.cols == other.rows,
		string.format(
			"Dimension mismatch in dot product: (%dx%d) vs (%dx%d)",
			self.rows,
			self.cols,
			other.rows,
			other.cols
		)
	)
	local res = Matrix.new(self.rows, other.cols)
	for r = 1, self.rows do
		local row_a = self.data[r]
		local row_res = res.data[r]
		for k = 1, self.cols do
			local val_a = row_a[k]
			if val_a ~= 0 then
				local row_b = other.data[k]
				for c = 1, other.cols do
					row_res[c] = row_res[c] + val_a * row_b[c]
				end
			end
		end
	end
	return res
end

function Matrix:transpose()
	local res = Matrix.new(self.cols, self.rows)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[c][r] = self.data[r][c]
		end
	end
	return res
end

function Matrix:add(other)
	local res = Matrix.new(self.rows, self.cols)
	if other.rows == 1 and other.cols == self.cols then
		for r = 1, self.rows do
			for c = 1, self.cols do
				res.data[r][c] = self.data[r][c] + other.data[c]
			end
		end
	else
		assert(self.rows == other.rows and self.cols == other.cols, "Matrix add dimension mismatch")
		for r = 1, self.rows do
			for c = 1, self.cols do
				res.data[r][c] = self.data[r][c] + other.data[r][c]
			end
		end
	end
	return res
end

function Matrix:sub(other)
	assert(self.rows == other.rows and self.cols == other.cols, "Matrix sub dimension mismatch")
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[r][c] = self.data[r][c] - other.data[r][c]
		end
	end
	return res
end

function Matrix:mul(other)
	assert(self.rows == other.rows and self.cols == other.cols, "Matrix mul dimension mismatch")
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[r][c] = self.data[r][c] * other.data[r][c]
		end
	end
	return res
end

function Matrix:div(other)
	assert(self.rows == other.rows and self.cols == other.cols, "Matrix div dimension mismatch")
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[r][c] = self.data[r][c] / other.data[r][c]
		end
	end
	return res
end

function Matrix:scale(scalar)
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[r][c] = self.data[r][c] * scalar
		end
	end
	return res
end

function Matrix:add_scalar(scalar)
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[r][c] = self.data[r][c] + scalar
		end
	end
	return res
end

function Matrix:sqrt()
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[r][c] = math.sqrt(self.data[r][c])
		end
	end
	return res
end

function Matrix:clip(min_val, max_val)
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			local v = self.data[r][c]
			if v < min_val then
				v = min_val
			end
			if v > max_val then
				v = max_val
			end
			res.data[r][c] = v
		end
	end
	return res
end

function Matrix:map(fn)
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[r][c] = fn(self.data[r][c], r, c)
		end
	end
	return res
end

function Matrix:sum_rows()
	local res = Matrix.new(1, self.cols, 0.0)
	for r = 1, self.rows do
		for c = 1, self.cols do
			res.data[c] = res.data[c] + self.data[r][c]
		end
	end
	return res
end

-- Leaky ReLU (prevents dying neurons)
function Matrix:leaky_relu(alpha)
	local a = alpha or 0.01
	return self:map(function(x)
		return x > 0 and x or (a * x)
	end)
end

function Matrix:leaky_relu_deriv(alpha)
	local a = alpha or 0.01
	return self:map(function(x)
		return x > 0 and 1.0 or a
	end)
end

-- Numerically stable Softmax
function Matrix:softmax()
	local res = Matrix.new(self.rows, self.cols)
	for r = 1, self.rows do
		local max_val = -math.huge
		for c = 1, self.cols do
			if self.data[r][c] > max_val then
				max_val = self.data[r][c]
			end
		end
		local sum_exp = 0.0
		for c = 1, self.cols do
			local e = math.exp(self.data[r][c] - max_val)
			res.data[r][c] = e
			sum_exp = sum_exp + e
		end
		for c = 1, self.cols do
			res.data[r][c] = res.data[r][c] / sum_exp
		end
	end
	return res
end

function Matrix:argmax_rows()
	local indices = {}
	for r = 1, self.rows do
		local best_c = 1
		local best_v = self.data[r]
		for c = 2, self.cols do
			if self.data[r][c] > best_v then
				best_v = self.data[r][c]
				best_c = c
			end
		end
		indices[r] = best_c
	end
	return indices
end

return Matrix
