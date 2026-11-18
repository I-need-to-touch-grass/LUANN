-- main.lua
-- Entry Point (v2): Adam + Cosine Annealing + Loss Curve + Interactive Drawing Pad

local Matrix = require("matrix")
local Dataset = require("dataset")
local Network = require("network")
local Serializer = require("serializer")
local Visualizer = require("visualizer")

local CONFIG = {
	train_samples = 3000,
	test_samples = 600,
	noise_level = 0.16,
	architecture = { 64, 64, 32, 10 },
	epochs = 40,
	batch_size = 32,
	lr_max = 0.008,
	lr_min = 0.0002,
	dropout = 0.15,
	l2_reg = 0.0004,
	model_path = "digit_model_v2.txt",
}

local function shuffle_samples(list)
	for i = #list, 2, -1 do
		local j = math.random(i)
		list[i], list[j] = list[j], list[i]
	end
end

-- Cosine Annealing Learning Rate Schedule
local function cosine_lr(epoch, max_epochs, lr_max, lr_min)
	local progress = (epoch - 1) / math.max(1, max_epochs - 1)
	return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
end

local function train_model()
	print("=======================================================")
	print("  PURE LUA DIGIT CLASSIFIER v2 (Adam + Dropout + Pad)  ")
	print("=======================================================")
	print(
		string.format(
			"Generating %d training & %d test samples (with rotation & cutout)...",
			CONFIG.train_samples,
			CONFIG.test_samples
		)
	)

	local train_data = Dataset.generate(CONFIG.train_samples, CONFIG.noise_level)
	local test_data = Dataset.generate(CONFIG.test_samples, CONFIG.noise_level)
	local X_test, Y_test, y_test_labels = Dataset.to_matrices(test_data, 1, #test_data)

	local net = Network.new(CONFIG.architecture, {
		lr = CONFIG.lr_max,
		dropout = CONFIG.dropout,
		l2_reg = CONFIG.l2_reg,
	})

	print(
		string.format(
			"Architecture: %s | Optimizer: Adam | Dropout: %.0f%%\n",
			table.concat(CONFIG.architecture, " -> "),
			CONFIG.dropout * 100
		)
	)

	local train_history = {}
	local test_history = {}
	local start_time = os.clock()

	for epoch = 1, CONFIG.epochs do
		shuffle_samples(train_data)
		net.lr = cosine_lr(epoch, CONFIG.epochs, CONFIG.lr_max, CONFIG.lr_min)

		local epoch_loss = 0.0
		local num_batches = 0

		for i = 1, #train_data, CONFIG.batch_size do
			local end_idx = math.min(i + CONFIG.batch_size - 1, #train_data)
			local X_batch, Y_batch = Dataset.to_matrices(train_data, i, end_idx)

			local probs = net:forward(X_batch, true)
			local loss = net:compute_loss(probs, Y_batch)
			net:backward(Y_batch)

			epoch_loss = epoch_loss + loss
			num_batches = num_batches + 1
		end

		local avg_train_loss = epoch_loss / num_batches
		local test_acc, test_loss = net:evaluate(X_test, Y_test, y_test_labels)

		train_history[epoch] = avg_train_loss
		test_history[epoch] = test_loss

		if epoch == 1 or epoch % 5 == 0 or epoch == CONFIG.epochs then
			print(
				string.format(
					"Epoch %2d/%d | Train Loss: %.4f | Test Loss: %.4f | Test Acc: %6.2f%% | LR: %.5f",
					epoch,
					CONFIG.epochs,
					avg_train_loss,
					test_loss,
					test_acc * 100,
					net.lr
				)
			)
		end
	end

	local elapsed = os.clock() - start_time
	print(string.format("\nTraining completed in %.2f seconds.", elapsed))

	local ok, err = Serializer.save(net, CONFIG.model_path)
	if ok then
		print("Saved trained weights to: " .. CONFIG.model_path)
	else
		print("Warning: " .. tostring(err))
	end

	local loaded_net = Serializer.load(CONFIG.model_path) or net
	local preds = loaded_net:predict(X_test)

	Visualizer.print_loss_curve(train_history, test_history)
	Visualizer.print_confusion_matrix(y_test_labels, preds)

	return loaded_net
end

local function interactive_menu(net)
	while true do
		print("Interactive Menu:")
		print("  [0-9] : Generate and classify a rotated/noisy digit")
		print("  [r]   : Classify a random noisy digit")
		print("  [d]   : Open the 8x8 Interactive Drawing Pad")
		print("  [q]   : Quit")
		io.write("Choice: ")

		local input = io.read("*l")
		if not input or input == "q" or input == "Q" then
			print("Exiting.")
			break
		elseif input == "d" or input == "D" then
			Visualizer.drawing_pad(net)
		elseif input == "r" or input == "R" then
			local digit = math.random(0, 9)
			local aug = Dataset.augment_grid(Dataset.get_prototype(digit), CONFIG.noise_level)
			local flat = Dataset.flatten(aug)
			local preds, _, probs = net:predict(Matrix.from_table({ flat }))
			Visualizer.render_prediction(flat, digit, preds, probs.data)
		else
			local digit = tonumber(input)
			if digit and digit >= 0 and digit <= 9 and math.floor(digit) == digit then
				local aug = Dataset.augment_grid(Dataset.get_prototype(digit), CONFIG.noise_level)
				local flat = Dataset.flatten(aug)
				local preds, _, probs = net:predict(Matrix.from_table({ flat }))
				Visualizer.render_prediction(flat, digit, preds, probs.data)
			else
				print("Invalid choice.\n")
			end
		end
	end
end

local net = train_model()
interactive_menu(net)
