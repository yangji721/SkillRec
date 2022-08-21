swig -c++ -python JobMatcher.i
g++ -fPIC -c JobMatcher.cpp
g++ -fPIC -c JobMatcher_wrap.cxx -I/home/comp/yangji/miniconda3/envs/SkillRec/include/python3.7m
g++ -shared JobMatcher.o JobMatcher_wrap.o -o _JobMatcherLinux.so