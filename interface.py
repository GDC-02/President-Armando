#CLI for inference
def main():
    print("Welcome to PresidentArmando, the best AI for Parkinson's risk prediction!")
    print("Please enter 9 comma-separated values as input, those value should be:\nMDVP:Jitter(Abs),\nMDVP:PPQ,\nMDVP:Shimmer,\nMDVP:APQ,\nNHR,\nRPDE,\nDFA,\nspread2,\nPPE:")

    while True:
        user_input = input("> ")

        try:
            # Convert input to tensor
            data = torch.tensor([float(x) for x in user_input.split(',')], dtype=torch.float32).view(1, -1)
            if data.shape[1] != 9:
                raise ValueError("Expected 9 input features, but got a different number.")

            # Predict
            PresidentArmando.eval()
            with torch.no_grad():
                prediction = PresidentArmando(data).item()  # .item() to get the scalar value
                if prediction<=0.03:
                    print(f"You have a low risk of contracting Parkinson's desease ({prediction:.4f})")
                elif prediction>0.03 and prediction<=0.3:
                    print(f"You have a moderate risk of contracting Parkinson's desease ({prediction:.4f})")
                elif prediction>0.3:
                    print(f"You have a high risk of contracting Parkinson's desease ({prediction:.4f})")

        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        #Ask if user wants to continue
        cont = input("Do you want to make another prediction? (yes/no): ").strip().lower()
        if cont=='yes':
            continue
        elif cont!='yes' and cont!='no':
            print("An error occurred. Please enter again the values on which you want to do a prediction (or random values if you want to stop predicting); then answer the next question only entering 'yes' or 'no', pay attention at any spelling mistake.")
            continue
        elif cont == 'no':
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
    
#Suggested tryout values: MDVP:Jitter(Abs):0.00007,MDVP:PPQ:0.00554,MDVP:Shimmer:0.04374,MDVP:APQ:0.02971,NHR:0.02211,RPDE:0.414783,DFA:0.815285,spread2:0.266482,PPE:0.284654
#0.00007,0.00554,0.04374,0.02971,0.02211,0.414783,0.815285,0.266482,0.284654
