:loop
start /WAIT "CMDchild" cmd /c "conda activate cotorch & python main_train_script.py"
echo "NaN Fail, Restart in 5"
timeout /t 5
goto loop
