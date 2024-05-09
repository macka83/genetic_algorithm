# genetic_algorithm

Inspired by David R Miller's work
https://github.com/davidrmiller/biosim4


-------------------
initial requirements  
Python 3.9.12  
conda 4.13.0  

Presented framework is my interpretation of biosim4 prepared by David R Miller and most of assumption come from his work. Great explanation is provided by D.R. Miller's video https://www.youtube.com/watch?v=N3tRFayqVtk. 

branch description:
  v_0 - working version before refactoring
  dev_0_semi_finished - 1st approach to refactor
  dev_1_classes_approach - 2nd approach to refactor. Applied black formatting and classes


The description applied to v_0

1) Creating initial population is almost the same as original except input and output neurons number to simplify calculations.
    * input neurons sensitive to  
        + input0 - close obstacle 1 point ahead)  
        + input1 - distant obstacle (up to 5 points ahead)  
    * output neurons move  
        + output0 - north  
        + output1 - south  
        + output2 - east  
        + output3 - west  
        + output4 - random movement 
2) in this experiment only asexual reproduction and punctual mutation are implemented

TODO:
1) improve main loop performance and increase ability to process larger population
2) small population which might give biased result in compariosn to larger population
3) add sexual reproducion
4) check if mutation works fine




