from asparagus import Asparagus

# Start training a default PhysNet model.
model = Asparagus(
    config='train.json',
    )
data = model.get_data_container()
calc = model.get_model_calculator()

results = calc.calculate_data(data)
print(results)

