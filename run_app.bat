@ECHO OFF

set root=C:\Dev\anaconda3
set project_path=D:\Projects\aml_final_proj
set env_name=new_env

call "%root%\Scripts\activate.bat"
cd %project_path%
call conda activate %env_name%
streamlit run app.py
