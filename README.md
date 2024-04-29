# Applied Cryptography - Project 2

## Project Title: Classifiying Encrypted Biometric Data for Facial Recognition & Authentication

**Project Type 4:** Designing a cryptography solution to allow computation of an outsourced algorithm or machine learning classifier over your encrypted input (test) data via homomorphic encryption scheme<br /><br />

**Team Name: Hard Core Bit-ches**<br /><br />

**Team Members:**<br />
 - Ariana Sutanto<br />
 - Mahmoud Shabana<br />
 - Reagan Bachman<br /><br />

**Design of Method/Solution:**<br />
- Import a PyTorch model <br />
- Import public facial recognition dataset<br />
- Create dataset with our faces and friend’s faces (will serve as authenticated list)<br />
- Run training on our model until we reach the desired level of accuracy (>93%)<br />
- Once training is complete and we have a fairly accurate model, deploy our model 
for inference<br />
- Prompt the user for an image input to be predicted against the model<br />
- Encrypt input using fully homomorphic encryption (FHE) implementation (reference 
implementation here)<br />
- Compile the model on encrypted input (possibly using an imported compiler like 
this)<br />
- Execute inference on encrypted input with the compiled model and save the results<br />
- ML classifier should identify the encrypted image against our dataset and determine 
if the identified individual is on our authentication list<br />
- Decrypt the result from a ciphertext to plaintext and return to the user<br />
- The objective is to maintain privacy once the user sends input from the client to the 
server (server is the deployed ML model in this case) and privacy through all 
intermediate steps. Once the computation is complete, the server returns the 
encrypted result to the client and the client decrypts the result<br />
- Evaluate our process by using different input sizes (large array of inputs) against an 
encrypted computation process vs a non-encrypted computation process<br />
