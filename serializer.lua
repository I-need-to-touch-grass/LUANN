-- serializer.lua
-- Save and Load Trained Neural Network Weights to Disk

local Matrix = require("matrix")
local Network = require("network")

local Serializer = {}

-- Save network topology and weights to a file
function Serializer.save(net, filepath)
	local file, err = io.open(filepath, "w")
	if not file then
		return false, "Failed to open file for writing: " .. tostring(err)
	end

	-- Header & Architecture
	file:write("LUA_MLP_V1\n")
	file:write(table.concat(net.layer_sizes, ",") .. "\n")

	-- Write each layer's weights and biases
	for i = 1, net.num_layers do
		local w = net.weights[i]
		file:write(string.format("W %d %d\n", w.rows, w.cols))
		for r = 1, w.rows do
			local row_strs = {}
			for c = 1, w.cols do
				row_strs[c] = string.format("%.8f", w.data[r][c])
			end
			file:write(table.concat(row_strs, " ") .. "\n")
		end

		local b = net.biases[i]
		file:write(string.format("B %d %d\n", b.rows, b.cols))
		local b_strs = {}
		for c = 1, b.cols do
			b_strs[c] = string.format("%.8f", b.data[1][c])
		end
		file:write(table.concat(b_strs, " ") .. "\n")
	end

	file:close()
	return true
end

-- Load network topology and weights from a file
function Serializer.load(filepath)
	local file, err = io.open(filepath, "r")
	if not file then
		return nil, "Failed to open file for reading: " .. tostring(err)
	end

	local magic = file:read("*l")
	if magic ~= "LUA_MLP_V1" then
		file:close()
		return nil, "Invalid model file format"
	end

	local arch_line = file:read("*l")
	local layer_sizes = {}
	for num in string.gmatch(arch_line, "%d+") do
		table.insert(layer_sizes, tonumber(num))
	end

	local net = Network.new(layer_sizes)

	for i = 1, net.num_layers do
		local w_header = file:read("*l")
		local wr, wc = string.match(w_header, "W (%d+) (%d+)")
		wr, wc = tonumber(wr), tonumber(wc)
		local w_mat = Matrix.new(wr, wc)

		for r = 1, wr do
			local line = file:read("*l")
			local c = 1
			for val in string.gmatch(line, "[^%s]+") do
				w_mat.data[r][c] = tonumber(val)
				c = c + 1
			end
		end
		net.weights[i] = w_mat

		local b_header = file:read("*l")
		local _, bc = string.match(b_header, "B (%d+) (%d+)")
		bc = tonumber(bc)
		local b_mat = Matrix.new(1, bc)
		local b_line = file:read("*l")
		local c = 1
		for val in string.gmatch(b_line, "[^%s]+") do
			b_mat.data[1][c] = tonumber(val)
			c = c + 1
		end
		net.biases[i] = b_mat
	end

	file:close()
	return net
end

return Serializer
