import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

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


def six_mussel_model(init_pop=20, recruitment=200, survival_rate=0.4, time_to_run=50):
    """
    Consider a mussel model in a fluctuating environment:
        X(t+1) = I(t) + S * X(t)
    where recruitment only occurs every second year so that:
        I(t) = 2I   if t is even
        I(t) = 0    if t is odd
    Prove that after any transient dynamics , the system ultimately executes two year cycles
    with:
        X(t) = 2IS/1-S^2    if t is even
        X(t) = 2I / 1-S^2   if t is odd

    :param init_pop: X(0), the starting population of the model
    :param recruitment: I(0), the recruitment to occur
    :param survival_rate: S, a small constant proportion of the population that survives
    :param time_to_run: max(t), the amount of time from 0 the model is to run
    """

    pop_history = []
    cycle_history = []
    current_pop = init_pop

    for t in range(0, time_to_run):
        # Here we calculate the recruitment value based on whether t is odd or even
        # Additionally, we also calculate what the population should be based on the two
        # year cycle shown above
        if t % 2 == 0:
            current_recruitment = 2 * recruitment
            cycle = (2 * recruitment * survival_rate) / (1 - math.pow(survival_rate, 2))
        else:
            current_recruitment = 0
            cycle = (2 * recruitment) / (1 - math.pow(survival_rate, 2))

        # This is the fluctuating environment calculation X(t+1) = I(t) + S * X(t)
        current_pop = current_recruitment + survival_rate * current_pop

        pop_history.append(current_pop)
        cycle_history.append(cycle)
        non_float_pop = round(current_pop, 2)
        non_float_cycle = round(cycle, 2)
        print("current_pop: {}, cycle: {}".format(non_float_pop, non_float_cycle))

    # Matplotlib work
    plt.style.use('seaborn-darkgrid')
    plt.tight_layout()

    plt.plot(
        range(0, time_to_run),
        pop_history,
        marker='',
        linewidth=1,
        alpha=0.9
    )

    plt.title("mussel model")
    plt.xlabel("time")
    plt.ylabel("pop")
    plt.show()

    # plt.plot(
    #     range(0, time_to_run),
    #     cycle_history,
    #     marker='',
    #     linewidth=1,
    #     alpha=0.9
    # )
    # plt.title("cycles")
    # plt.xlabel("time")
    # plt.ylabel("cycle")
    # plt.show()


def project_continuous_time_logistic_model(init_pop=400, time_to_run=50, r=0.4, k=300):
    """
    dN/dt = rN(1-N/K)
    N(t+1) = Nt + dN/dt
           = Nt + rN(1 - N/K)
    """
    current_pop = init_pop
    pop_history = []
    for i in range(0, time_to_run):
        current_pop = current_pop + r * current_pop * (1 - current_pop/k)
        pop_history.append(current_pop)

    # Matplotlib work
    plt.style.use('seaborn-darkgrid')
    plt.tight_layout()

    plt.plot(
        range(0, time_to_run),
        pop_history,
        marker='',
        linewidth=1,
        alpha=0.9
    )

    plt.title("mussel model")
    plt.xlabel("time")
    plt.ylabel("pop")
    plt.show()


def project_continuous_time_logistic_model_sinusoid(init_pop=20, time_to_run=50, r=0.4, k_0=200, k_1=20, t_p = 10):
    """
      dN/dt = rN(1-N/K)
      N(t+1) = Nt + dN/dt
             = Nt + rN(1 - N/K)
      K varies sinusoidally with time
    """

    current_pop = init_pop
    pop_history = []
    for t in range(0, time_to_run):
        current_k = k_0 + k_1 * math.cos(2 * math.pi * t / t_p)
        current_pop = current_pop + r * current_pop * (1 - current_pop / current_k)
        pop_history.append(current_pop)

    # Matplotlib work
    plt.style.use('seaborn-darkgrid')
    plt.tight_layout()

    plt.plot(
        range(0, time_to_run),
        pop_history,
        marker='',
        linewidth=1,
        alpha=0.9
    )

    plt.title("mussel model")
    plt.xlabel("time")
    plt.ylabel("pop")
    plt.show()


if __name__ == "__main__":
    # one_exponential_growth()
    # two_logistic_map()
    # mussel_model()
    # continuous_time_logistic_model()
    project_continuous_time_logistic_model_sinusoid()
