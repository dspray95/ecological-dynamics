import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def one_exponential_growth(init_pop=2, growth_factor=12, time=8):
    """
    p(0) = exp(r) * p(t)
    r = ln(p(t)) - ln(p(0)/t
    r = ln(p(0) * 12 - ln(p(0))/t
    :return:
    """
    pop_t = growth_factor * time
    r = (np.log(pop_t) - np.log(init_pop)) / time
    print("r value is: {}".format(r))
    grow_by_one_million = np.log(1000000)/r
    print("time to grow by a factor of one million: {}".format(grow_by_one_million))


def two_logistic_map(init_pop=2, b_init=3.1, b_range=0.8, c=0.001, b_increment=0.2, time_to_run=50):
    """
    nt+1 = bnt(1-cnt)

    :return:
    """
    results = pd.DataFrame({'x': range(0, time_to_run)})
    b_values = []  # b values are stored as keys for the data frame
    b_max = b_init + b_range  # Limit for b
    b = b_init - b_range  # initially set b to lowest possible b value
    # Iterate through various possible values of b, running population models with each value and storing them
    # with the relative b value as its key in the results data frame.
    while b <= b_max + (b_increment / 2):
        pop_history = []
        current_pop = init_pop
        for t in range(0, time_to_run):
            next_pop = (b * current_pop) * (1 - c * current_pop)  # Actual equation here
            pop_history.append(next_pop)
            current_pop = next_pop
        results['{0:.1f}'.format(b)] = pop_history  # Formatting here is just cutting b off after one decimal place

        b_values.append('{0:.1f}'.format(b))
        b += b_increment

    # Matplotlib setup
    plt.style.use('seaborn-darkgrid')
    plt.tight_layout()

    plt_num = 0  # keeps track of our current plot
    for column in results.drop('x', axis=1):
        plt_num += 1
        plt.subplot(3, 3, plt_num)
        # We used b as the key, so we need to get the values again to pull the list from the data frame
        plt.plot(
            results['x'],
            results[b_values[plt_num - 1]],
            marker='',
            linewidth=1,
            alpha=0.9, label=b_values[plt_num - 1]
        )

        # Make sure the limits are the same across each graph
        largest_value = int(results.values.max())
        largest_value -= largest_value % -100  # modulo hack to round up to nearest 100
        plt.ylim(0, largest_value)
        plt.xlim(0, time_to_run)

        # finally sort the labels out
        plt.title(b_values[plt_num - 1], loc='left', fontsize=12, fontweight=0)
        plt.xlabel("time")
        plt.ylabel("pop")

    plt.show()


def three_lake_toxicant():

    pass


if __name__ == "__main__":
    one_exponential_growth()
    two_logistic_map()
