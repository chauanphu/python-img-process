from subprocess import call

def set_volume(value=100):
    call(["amixer", "-D", "pulse", "sset", "Master", f"{value}%"])