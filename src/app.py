from genre_satisfaction import main as satisfaction_main
from movies import main as distribution_main
from ratings_over_time import main as ratings_over_time_main

for func in (distribution_main, satisfaction_main, ratings_over_time_main):
    try:
        func()
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
    print()
