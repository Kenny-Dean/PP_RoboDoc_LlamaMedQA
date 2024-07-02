# Erstellen einer neuen virtuellen Umgebung
python3.10 -m venv robodoc_llama_train

# Aktivieren der virtuellen Umgebung
source robodoc_llama_train/bin/activate

# Installieren der requirements
pip install -r requirements.txt

# Erstellen von einer detachable Session 
tmux new-session -d
tmux ls
tmux attach -t "session_id"

Mit Crtl + b und d kann sich von der Session getrennt werden und das Training läuft weiter.
Ein checken des Standes ist dann wieder mit tmux attach -t "session_id" möglich.




(Möglicherweise erneutes gehen in das Python env)
# Aktivieren der virtuellen Umgebung
source robodoc_llama_train/bin/activate


# Ausführen des Scriptes
python3.10 training.py


Sonstiges:

Torchversion ist die kompatible mit der CUDA 12.1 Version

Trainiert wurde mit CUDA 12.1, falls eine andere Version hinterlegt ist (und Cuda 12.1 installiert aber nicht als "Hauptversion" hinterlegt ist.) 

Helfen folgende Commands:

export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

