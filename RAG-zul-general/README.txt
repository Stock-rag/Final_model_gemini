requires python 3.10 or transformers package is not working for newer version

STEP-1 :try python -- version 
should show 3.10 if not download and install it and add the path 
STEP-2 if 3.13 is also still there then 
# Replace the path with your Python 3.10 path
"C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python310\python.exe" -m venv rag_env_310

this will create an isolated enviorment where you can add all your packages specific to the project

or if version is 3.10 : python -m venv {enviorment_name}
this will create a folder in current directory then
{enviorment_name}\Scripts\activate 
this will activate the enviorment you will see {enviorment_name} infront of you command line
now run python files here to use the isolated packages in here 
you can use pip install to install packages 
Step -2 pip install transformers langchain
Step -3 python hugging_face_llm.py 

after done use deactivate to come out of the venv
