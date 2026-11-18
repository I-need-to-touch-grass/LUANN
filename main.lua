-- main.lua
-- Entry Point: Train, Evaluate, Save, and Test the Lua Digit Classifier

local Matrix = require("matrix")
local Dataset = require("dataset")
local Network = require("network")
local Serializer = require("serializer")
local Visualizer = require("visualizer")

local CONFIG = {
	train_samples = 2000,
	test_samples = 500,
	noise_level = 0.18,
	architecture = { 64, 48, 24, 10 },
	epochs = 35,
	batch_size = 32,
	lr = 0.06,
	lr_decay = 0.96,
	momentum = 0.9,
	l2_reg = 0.0004,
	model_path = "digit_model.txt",
}

local function shuffle_samples(list)
	for i = #list, 2, -1 do
		local j = math.random(i)
		list[i], list[j] = list[j], list[i]
	end
end

local function train_model()
	print("====================================================")
	print("   PURE LUA HANDWRITTEN DIGIT CLASSIFIER (8x8 MLP)  ")
	print("====================================================")
	print(string.format("Generating %d training & %d test samples...", CONFIG.train_samples, CONFIG.test_samples))

	local train_data = Dataset.generate(CONFIG.train_samples, CONFIG.noise_level)
	local test_data = Dataset.generate(CONFIG.test_samples, CONFIG.noise_level)

	local X_test, Y_test, y_test_labels = Dataset.to_matrices(test_data, 1, #test_data)

	local net = Network.new(CONFIG.architecture, {
		lr = CONFIG.lr,
		momentum = CONFIG.momentum,
		l2_reg = CONFIG.l2_reg,
	})

	print(
		string.format(
			"Architecture: %s | Epochs: %d | Batch Size: %d\n",
			table.concat(CONFIG.architecture, " -> "),
			CONFIG.epochs,
			CONFIG.batch_size
		)
	)

	local start_time = os.clock()

	for epoch = 1, CONFIG.epochs do
		shuffle_samples(train_data)
		local epoch_loss = 0.0
		local num_batches = 0

		for i = 1, #train_data, CONFIG.batch_size do
			local end_idx = math.min(i + CONFIG.batch_size - 1, #train_data)
			local X_batch, Y_batch = Dataset.to_matrices(train_data, i, end_idx)

			local probs = net:forward(X_batch)
			local loss = net:compute_loss(probs, Y_batch)
			net:backward(Y_batch)

			epoch_loss = epoch_loss + loss
			num_batches = num_batches + 1
		end

		-- Decay learning rate each epoch
		net.lr = net.lr * CONFIG.lr_decay

		if epoch == 1 or epoch % 5 == 0 or epoch == CONFIG.epochs then
			local test_acc, test_loss = net:evaluate(X_test, Y_test, y_test_labels)
			print(
				string.format(
					"Epoch %2d/%d | Train Loss: %.4f | Test Loss: %.4f | Test Acc: %6.2f%% | LR: %.4f",
					epoch,
					CONFIG.epochs,
					epoch_loss / num_batches,
					test_loss,
					test_acc * 100,
					net.lr
				)
			)
		end
	end

	local elapsed = os.clock() - start_time
	print(string.format("\nTraining completed in %.2f seconds.", elapsed))

	-- Save trained model
	local ok, err = Serializer.save(net, CONFIG.model_path)
	if ok then
		print("Saved trained weights to: " .. CONFIG.model_path)
	else
		print("Warning: " .. tostring(err))
	end

	-- Reload to verify serialization integrity
	local loaded_net = Serializer.load(CONFIG.model_path) or net
	local preds, _, probs = loaded_net:predict(X_test)
	Visualizer.print_confusion_matrix(y_test_labels, preds)

	return loaded_net, test_data, probs
end

local function interactive_loop(net)
	print("Interactive Mode:")
	print("  - Enter a digit (0-9) to generate a noisy sample and classify it")
	print("  - Enter 'r' for a random noisy digit")
	print("  - Enter 'q' to quit\n")

	while true do
		io.write("Command [0-9 / r / q]: ")
		local input = io.read("*l")
		if not input or input == "q" or input == "Q" then
			print("Exiting.")
			break
		end

		local digit
		if input == "r" or input == "R" then
			digit = math.random(0, 9)
		else
			digit = tonumber(input)
		end

		if digit and digit >= 0 and digit <= 9 and math.floor(digit) == digit then
			local proto = Dataset.get_prototype(digit)
			local aug = Dataset.augment_grid(proto, CONFIG.noise_level)
			local flat = Dataset.flatten(aug)
			local X_single = Matrix.from_table({ flat })

			local preds, _, probs = net:predict(X_single)
			Visualizer.render_prediction(flat, digit, preds[1], probs.data[1])
		else
			print("Invalid input. Please enter 0-9, 'r', or 'q'.")
		end
	end
end

-- Run pipeline
local net, test_data, test_probs = train_model()

print("Sample Test Predictions:")
for i = 1, 3 do
	local sample = test_data[i]
	local X_s = Matrix.from_table({ sample.x })
	local preds = net:predict(X_s)
	Visualizer.render_prediction(sample.x, sample.label, preds[1], test_probs.data[i])
end

interactive_loop(net)
