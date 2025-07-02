import random
import os

try:
    import wandb
    print("wandb imported successfully")
except ImportError:
    print("wandb not installed. Installing...")
    os.system("pip install wandb")
    import wandb

# 设置离线模式（如果没有wandb账号）
os.environ["WANDB_MODE"] = "offline"

try:
    # Start a new wandb run to track this script.
    run = wandb.init(
        # 使用默认entity，或者设置为你的用户名
        entity=None,  # 或者 "your-username"
        # 设置项目名称
        project="heatnn-aurora-experiments",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    # Simulate training.
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        # Log metrics to wandb.
        run.log({"acc": acc, "loss": loss})
        print(f"Epoch {epoch}: acc={acc:.4f}, loss={loss:.4f}")

    # Finish the run and upload any remaining data.
    run.finish()
    print("wandb run completed successfully!")

except Exception as e:
    print(f"Error running wandb: {e}")
    print("You may need to:")
    print("1. Run 'wandb login' to authenticate")
    print("2. Or set WANDB_MODE=offline for offline mode")
    print("3. Or install wandb: pip install wandb") 