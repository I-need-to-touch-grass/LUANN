-- dataset.lua
-- 8x8 Handwritten Digit Dataset Generator with Data Augmentation
-- Comment: Without an External basic Machine learning dataset handcoding values like this is hard so i used AI only to generate local PROTOTYPES:w

local Matrix = require("matrix")
local Dataset = {}

-- Base 8x8 pixel prototypes for digits 0-9 (values 0.0 to 1.0)
local PROTOTYPES = {
	[0] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[1] = {
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 1, 0, 0, 0 },
		{ 0, 1, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
	},
	[2] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
	},
	[3] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 1, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[4] = {
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 1, 1, 1, 0, 0 },
		{ 0, 0, 1, 0, 1, 1, 0, 0 },
		{ 0, 1, 0, 0, 1, 1, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
	},
	[5] = {
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[6] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[7] = {
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
	},
	[8] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[9] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 1, 1, 1, 0, 0, 0 },
	},
}

-- Clamp utility
local function clamp(val, min_v, max_v)
	if val < min_v then
		return min_v
	end
	if val > max_v then
		return max_v
	end
	return val
end

-- Apply random translation (-1, 0, +1 px), stroke jitter, and Gaussian-like noise
function Dataset.augment_grid(grid, noise_level)
	local shift_r = math.random(-1, 1)
	local shift_c = math.random(-1, 1)
	local intensity_scale = 0.8 + math.random() * 0.4
	local out = {}

	for r = 1, 8 do
		out[r] = {}
		for c = 1, 8 do
			local src_r = r - shift_r
			local src_c = c - shift_c
			local base = 0.0
			if src_r >= 1 and src_r <= 8 and src_c >= 1 and src_c <= 8 then
				base = grid[src_r][src_c] * intensity_scale
			end
			-- Random pixel dropout or bleed
			if base > 0.5 and math.random() < 0.08 then
				base = base * 0.4
			elseif base == 0 and math.random() < 0.05 then
				base = 0.25
			end
			-- Add uniform sensor noise
			local noise = (math.random() * 2.0 - 1.0) * (noise_level or 0.15)
			out[r][c] = clamp(base + noise, 0.0, 1.0)
		end
	end
	return out
end

-- Flatten an 8x8 grid into a 64-element 1D array
function Dataset.flatten(grid)
	local flat = {}
	local idx = 1
	for r = 1, 8 do
		for c = 1, 8 do
			flat[idx] = grid[r][c]
			idx = idx + 1
		end
	end
	return flat
end

-- Generate a dataset of N samples with features (N x 64) and one-hot labels (N x 10)
function Dataset.generate(num_samples, noise_level)
	local samples = {}
	for i = 1, num_samples do
		local digit = (i - 1) % 10
		local aug = Dataset.augment_grid(PROTOTYPES[digit], noise_level)
		local flat = Dataset.flatten(aug)
		local one_hot = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
		one_hot[digit + 1] = 1.0
		samples[i] = {
			x = flat,
			y = one_hot,
			label = digit,
		}
	end

	-- Fisher-Yates shuffle
	for i = #samples, 2, -1 do
		local j = math.random(i)
		samples[i], samples[j] = samples[j], samples[i]
	end

	return samples
end

-- Convert a slice of samples into Matrix X (batch x 64) and Matrix Y (batch x 10)
function Dataset.to_matrices(samples, start_idx, end_idx)
	local count = end_idx - start_idx + 1
	local X = Matrix.new(count, 64)
	local Y = Matrix.new(count, 10)
	local labels = {}

	for i = 1, count do
		local s = samples[start_idx + i - 1]
		X.data[i] = s.x
		Y.data[i] = s.y
		labels[i] = s.label
	end

	return X, Y, labels
end

function Dataset.get_prototype(digit)
	return PROTOTYPES[digit]
end

return Dataset
