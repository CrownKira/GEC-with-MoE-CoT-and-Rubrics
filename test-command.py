import subprocess

# Your multi-line string
text = """Fer
we think that in the future the planet will be in bad conditions and the trees will be disappearing, after that we will be having wars.
In 30 years we will have changed our anatomy, also we will be eating fast food, on the other hand, the north pole will have melted totally.
The temperature will have become crazy by global warming, so some people will have died because the natural disasters will be more aggressive.
The Technology will have advanced and maybe the cars will be flying by streets and computers will have totally changed.
Because of this, we have to raise awareness of what is happening and we help the planet.
Friendship is something very important in my life.
I can't imagine my lifetime without friends.
How to make friends and meet new people?
It is easier than you think.
Just ... start talking!
Communication is the most important point when you're going to make friends.
You have to remember, that friends are not supposed to agree on every single thing.
They just have to calm talk about it.
If your friendship is real, you will always find point between your opinion and your friend's one.
Just try, it won't cost you much!"""

# Pass the multi-line string to a bash command
# Here, we're using 'echo' as an example command
# Note: 'shell=True' is generally not recommended due to security considerations, especially with untrusted input
subprocess.run(["python3", "-m", "systems.greco", f'"{text}"'], check=True)
