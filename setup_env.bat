@ECHO OFF

set root=C:\Dev\anaconda3
set project_path=D:\Projects\aml_final_proj
set env_name=new_env

call "%root%\Scripts\activate.bat"
cd %project_path%

rem Check if the environment exists, create if not
conda info --envs | find "%env_name%" > nul
if %errorlevel% neq 0 (
    call conda create -n %env_name% --file .\requirements.txt
) else (
    call conda activate %env_name%
)

@ECHO Setup has been successful!
timeout /t 5
@ECHO Exiting the script
