#Initialize seed for stability
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

#Group by class
groups = [group for _, group in df_n.groupby('status')]

#Split each group into train and test sets
train_groups = []
test_groups = []

for group in groups:
    #Shuffle each group
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(group))
    
    #Append split data
    train_groups.append(group.iloc[:split_idx])
    test_groups.append(group.iloc[split_idx:])

#Combine the groups
train_df = pd.concat(train_groups).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_groups).sample(frac=1, random_state=42).reset_index(drop=True)

#Separate features and targets
X_train, y_train = train_df.drop(columns='status'), train_df['status']
X_test, y_test = test_df.drop(columns='status'), test_df['status']

print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")

#Convert data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  #Make target column vector
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#Defining the Neural Network
class Ann(nn.Module):
    def __init__(self):
        super(Ann, self).__init__()
        #First hidden layer (9 input features to 5 neurons)
        self.linear1 = nn.Linear(9, 5)
        #Activation function after first hidden layer
        self.relu1 = nn.ReLU()

        #Second hidden layer (5 neurons to 5 neurons)
        self.linear2 = nn.Linear(5,5)
        # Activation function after second hidden layer
        self.relu2 = nn.ReLU()

        #Output layer (5 neurons to 1 output neuron)
        self.linear3 = nn.Linear(5, 1)

    def forward(self, x):
        #Pass through first layer and activation
        x = self.linear1(x)
        x = self.relu1(x)
        #Pass through second layer and activation
        x = self.linear2(x)
        x = self.relu2(x)
        #Pass through output layer
        x = self.linear3(x)
        return x
        
#Define an instance of the model
PresidentArmando = Ann()

#Loss function and optimizer
loss_f = nn.MSELoss()
optimizer = optim.SGD(PresidentArmando.parameters(), lr=0.01)

#Loss visualization setup
losses = []
accuracies=[]

#Training
for epoch in range(50000):
    PresidentArmando.train()
    predictions = PresidentArmando(X_train)
    loss = loss_f(predictions, y_train)
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  # Calculate accuracy every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        with torch.no_grad():
            pred_labels = (PresidentArmando(X_train) > 0.5).float()
            accuracy = (pred_labels == y_train).float().mean().item()
            accuracies.append(accuracy)
            if (epoch+1)%10000==0:
                print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}")
            
 # Plotting accuracy every 1000 epochs
plt.plot(range(1000, 50001, 1000), accuracies, marker='o')
plt.title("Graph E: Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()  
    
#Plotting losses over epochs
plt.plot(range(1, 50001), losses)
plt.title("Graph F: Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

#Evaluation
PresidentArmando.eval()
with torch.no_grad():
    test_predictions = PresidentArmando(X_test)
    test_loss = loss_f(test_predictions, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

    # Calculate test accuracy
    test_pred_labels = (test_predictions > 0.5).float()
    test_accuracy = (test_pred_labels == y_test).float().mean().item()
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Confusion matrix
    y_test_np = y_test.numpy().flatten()
    test_pred_labels_np = test_pred_labels.numpy().flatten()
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_test_np, test_pred_labels_np):
        cm[int(t), int(p)] += 1

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("Graph G: Confusion Matrix")
    fig.colorbar(cax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Sick"])
    ax.set_yticklabels(["Healthy", "Sick"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='red')

    plt.show()
    
    #Compare predictions with actual data
    print(f"Predictions: {test_predictions[:5].squeeze().detach().numpy()}")
    print(f"Actual: {y_test[:5].squeeze().numpy()}")
