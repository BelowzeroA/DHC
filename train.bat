set PYTHONPATH=%PYTHONPATH%;d:\Work\Projects\PyDev\DHC\lib
rem del data\handled_doctors.txt
del data\submission.csv
del data\temp_output.txt
FOR /L %%x IN (1,1,15) DO (
d:\python\python.exe trials\data_test.py
)