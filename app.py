from genre_satisfaction import main as satisfaction_main
from movies import main as distribution_main
from ratings_over_time import main as ratings_over_time_main

if __name__ == "__main__":
    distribution_main()
    print()
    satisfaction_main()
    print()
    ratings_over_time_main()
