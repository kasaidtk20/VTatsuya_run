import subprocess as sp
from main import main

wmc = r"C:\Users\m0727_d73jsnl\Downloads\PCsoft\WebcamMotionCapture_Win_ja\WebcamMotionCapture.exe"
VScode = r"C:\Users\m0727_d73jsnl\AppData\Local\Programs\Microsoft VS Code\Code.exe"
#obs = r"C:\Program Files\obs-studio\bin\64bit"

wmc_s = sp.Popen(wmc)
VScode_s = sp.Popen([VScode, "main.py"])
main()
#obs_s = sp.run(["start", "obs64.exe"], shell=True)
#obs_s = sp.run("cd", shell=True, cwd=obs)
#sp.run(["start", "\OBS Studio (64bit).exe"], shell=True)


#input("何かを入力し、Enterを押すとWMC,VScodeを終了します。：")
#wmc_s.kill()
#VScode_s.kill()
#obs_s.kill()