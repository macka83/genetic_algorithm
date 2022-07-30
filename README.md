# genetic_algorithm

Inspired by David R Miller's work
https://github.com/davidrmiller/biosim4
https://www.youtube.com/watch?v=N3tRFayqVtk

-------------------
requirements
Python 3.9.12
conda 4.13.0

Presented framework is my interpretation of biosim4 prepared by David R Miller and most of assumption come from his work

Componenets:
1) inital population
2) movement with respect to inout neurons  
    * input neurons sensitive to  
        + close obstacle (1 point ahead)  
        + distant obstacle (up to 5 points ahead)  
    * output neurons move  
        + north  
        + south  
        + east  
        + west  
        + randomly  
    
3) select individuals from safe zone
4) asexual reproduction + mutation

