from accelerate import Accelerator
import torch

accelerator = Accelerator()

# Example: Simple training loop
model = torch.nn.Linear(10, 10)  # Example model
optimizer = torch.optim.Adam(model.parameters())

# Move model to the appropriate device
model, optimizer = accelerator.prepare(model, optimizer)

for epoch in range(10):
    # Example data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 10)
    
    # Training step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
