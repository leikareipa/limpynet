# limpynet
A simple console-based feedforward neural net with customizable layering. Learns the MNIST digits to about 97%, depending on settings.

It's worth noting that you wouldn't really use this program for serious and/or performant neural netting; it's only something fun I messed around with. Recurrent neural nets would be a bit more interesting still, but a bit more complicated, too, for implementing from the ground up.

There's no way to save a trained net's weights. But as a consolation, once training is finished, you get the quiz mode: digits are randomly drawn from the MNIST validation set and into the console, followed by a display of whether the net correctly identifies that digit.

Note that you need to obtain and extract the MNIST database into a ```mnist``` folder subject to where you placed the limpynet executable.

## Command line
The following options are available on the command line.
- ```-R n``` Add a new layer of n neurons with a relu activation function.
- ```-L n``` Add a new layer of n neurons with a leaky relu activation function.
- ```-T n``` Add a new layer of n neurons with a tanh activation function.
- ```-G n``` Add a new layer of n neurons with a log activation function.
- ```-e n``` Set the number of training epochs. An epoch consists of x samplings of the training database, where x is the size of the database.
- ```-x``` Run a XOR diagnostic. The result should always be 100%. If it's not, there may be an issue with the network.
- ```-r x``` Set the learning rate to x; which might generally be a value of 0.1 to 0.0001.

## Sample output
```
$ ./limpynet -L 10 -e 3
Initializing for MNIST...
Net:	Topology: N784-L10-S10 
	Learning rate: 0.010000
	Training epochs: 3
Training on MNIST (60000/10000)...
Epoch 1 of 3: train = 68.947%, validate = 0.000%.
Epoch 2 of 3: train = 90.213%, validate = 90.300%.
Epoch 3 of 3: train = 91.570%, validate = 90.770%.
Training finished.
<Press enter to start the quiz, or CTRL+C to quit.>
                            
                            
                .***####*   
              .*##**....    
          .*#####.          
         .##**#.            
        .#*                 
        .*                  
        **                  
        #*                  
        *#*.                
        .*##**.             
          ..###*.           
             .*##*.         
               .*#*         
                 .#*        
      ..          .#.       
      .#          .#.       
       #*.        *#.       
       .##*.     .#*        
         *###**####.        
            .**#**          
                            
                            
The net guesses 5. That's correct.
```
