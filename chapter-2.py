import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

DIVIDER = "========================="


def one_exponential_growth(growth_factor=12, time=8):
    """
    to work out r value given a growth factor and time

    X(t) = X(0)exp(rt)
    X(t = time) = growth_factor * X

    r = ln(12X(0) - ln(X(0)) / t
    when t = 8
    r = ln(12X(0) - ln(X(0)) / 8 = ln(12) / 8

    then to work out time to growth factor of 1 million
    t = ln(10^6)/r
    """
    r_value = math.log(growth_factor) / time
    time_to_one_million = math.log(math.pow(10, 6)) / r_value

    print(DIVIDER)
    print("Q1: r value = {}, time to one million growth factor = {} ".format(r_value, time_to_one_million))


def two_logistic_map(init_pop=2, b_init=3.1, b_range=0.8, c=0.001, b_increment=0.2, time_to_run=50):
    """
    Using logistic map:
        n(t+1) = b * n(t)(1- c * n(t))


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
        pop_history.append(current_pop)
        for t in range(1, time_to_run):
            next_pop = (b * current_pop) * (1 - c * current_pop)  # Actual equation here
            pop_history.append(next_pop)
            current_pop = next_pop
        results['{0:.1f}'.format(b)] = pop_history  # Formatting here is just cutting b off after one decimal place

        b_values.append('{0:.1f}'.format(b))
        b += b_increment

    # Matplotlib setup
    plt.style.use('seaborn-paper')
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
        plt.title("b value = {}".format(b_values[plt_num - 1]), loc='left', fontsize=10, fontweight=0)
        plt.xlabel("time")
        plt.ylabel("pop")

    plt.savefig("graphs/chapter-2/q2-logistic-map-{}-{}-{}-{}-{}-{}.png".format(init_pop, b_init, b_range, c, b_increment, time_to_run))
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
        !! Possibly a misprint in the book? X(t) ends up being 2I/1-S^2 when t is even and 2IS/1-S^2 when t is odd !!
        !! In this repository the correct cycles are used !!

    :param init_pop: X(0), the starting population of the model
    :param recruitment: I(0), the recruitment to occur
    :param survival_rate: S, a small constant proportion of the population that survives
    :param time_to_run: max(t), the amount of time from 0 the model is to run
    """
    print(DIVIDER)
    print("Q6 Mussel Model")

    pop_history = []
    cycle_history = []
    current_pop = init_pop
    # Append population for t = 0 here, as the formula only calculates for pop at t+1, meaning t=0 is never recorded
    pop_history.append(current_pop)
    cycle_history.append((2 * recruitment) / (1 - math.pow(survival_rate, 2)) )
    for t in range(1, time_to_run):
        # Here we calculate the recruitment value based on whether t is odd or even
        # Additionally, we also calculate what the population should be based on the two
        # year cycle shown above
        if t % 2 == 0:
            current_recruitment = 2 * recruitment
            cycle = (2 * recruitment) / (1 - math.pow(survival_rate, 2))  # 2I / 1 - S^2
        else:
            current_recruitment = 0
            cycle = (2 * recruitment * survival_rate) / (1 - math.pow(survival_rate, 2))  # 2IS/ 1 - S^2

        # This is the fluctuating environment calculation X(t+1) = I(t) + S * X(t)
        current_pop = current_recruitment + (survival_rate * current_pop)

        pop_history.append(current_pop)
        cycle_history.append(cycle)
        non_float_pop = round(current_pop, 2)
        non_float_cycle = round(cycle, 2)
        print("current_pop: {}, cycle: {}".format(non_float_pop, non_float_cycle))

    # Matplotlib work
    plt.style.use("seaborn-paper")
    # plt.tight_layout()

    x = range(0, time_to_run)
    plt.plot(
        x,
        pop_history,
        marker='',
        linewidth=1,
        alpha=0.9,
        label="population"
    )

    plt.plot(
        x,
        cycle_history,
        marker='o',
        linestyle="",
        linewidth=1,
        label="two-year cycles"
    )

    plt.title("Mussel Model")
    plt.xlabel("time")
    plt.ylabel("pop")
    plt.legend(loc='upper left', frameon=True)
    if init_pop == 20 and recruitment == 200 and survival_rate == 0.4 and time_to_run == 50:
        plt.savefig("graphs/chapter-2/q6-mussel-model-default.png")
    else:
        plt.savefig("graphs/chapter-2/q6-mussel-model-{}-{}-{}-{}.png".format(init_pop, recruitment, survival_rate,
                                                                              time_to_run))
    plt.show()


def project_continuous_time_logistic_model(init_pop=20, time_to_run=50, r=0.4, k=300,
                                           sinusoid=False, sinusoid_k0=100, sinusoid_k1=10, sinusoid_tp=10,
                                           modify_k_values=False, modify_initial_pop=False, modify_tp_value=False):
    """"""

    results = pd.DataFrame({'x': np.linspace(0, time_to_run, time_to_run * 2)})

    if sinusoid and modify_initial_pop and sinusoid_k1 + (10 * 100) < init_pop + (10 * 100) and not modify_k_values:
        raise AttributeError("If modifying the initial population of and using the sinusoid function, the k values"
                             "must also be increased, otherwise the population will drop below zero")
    if sinusoid and sinusoid_k0 < sinusoid_k1:
        raise AttributeError("Sinusoid_k0({}) must be larger than sinusoid_k1({})".format(sinusoid_k0, sinusoid_k1))

    for i in range(1, 10):  # 1-10 rather than 0-9 to avoid divide by zero errors
        pop_history = []
        current_k = k

        if modify_initial_pop:
            current_pop = init_pop + i * 100
        else:
            current_pop = init_pop

        if modify_tp_value:
            tp = sinusoid_tp + i * 10
        else:
            tp = sinusoid_tp

        for t in np.linspace(0, time_to_run, time_to_run * 2):
            if sinusoid:
                if modify_k_values:
                    k_0 = sinusoid_k0 + i * 100
                    k_1 = sinusoid_k1 + i * 100
                else:
                    k_0 = sinusoid_k0
                    k_1 = sinusoid_k1

                current_k = k_0 + k_1 * math.cos(2 * math.pi * t / tp)
            else:
                if modify_k_values:
                    current_k = k + i * 100
                else:
                    current_k = k

            current_pop = current_pop + r * current_pop * (1 - current_pop / current_k)
            if current_pop < 0 or current_pop == math.inf:
                print("Pop dropped to an invalid number, check params!")
            pop_history.append(current_pop)

        set_label = "i:{}, k:{}".format(i, current_k)
        results[set_label] = pop_history

    # matplotlib work
    plt.style.use("seaborn-paper")
    plt.tight_layout()

    plt_num = 0  # keeps track of our current plot
    for column in results.drop('x', axis=1):
        plt_num += 1
        plt.subplot(3, 3, plt_num)
        # We used b as the key, so we need to get the values again to pull the list from the data frame
        plt.plot(
            results['x'],
            results[column],
            marker='',
            linewidth=1,
            alpha=0.9, label=results[column]
        )

        # Make sure the limits are the same across each graph
        largest_value = int(results.values.max())
        largest_value -= largest_value % -100  # modulo hack to round up to nearest 100
        largest_value += largest_value / 10
        plt.ylim(0, largest_value)
        plt.xlim(0, time_to_run)

        # finally sort the labels out
        # plt.title(results[str(plt_num * k)], loc='left', fontsize=12, fontweight=0)
        plt.xlabel("time")
        plt.ylabel("pop")

    plt.show()


if __name__ == "__main__":
    # one_exponential_growth()
    # two_logistic_map()
    # two_logistic_map(init_pop=250)
    # six_mussel_model()
    # six_mussel_model(init_pop=800, recruitment=500, survival_rate=0.1)

    # continuous_time_logistic_model()
    # project_continuous_time_logistic_model_sinusoid()
    # project_continuous_time_logistic_model()
    project_continuous_time_logistic_model()
    project_continuous_time_logistic_model(modify_initial_pop=True)
    project_continuous_time_logistic_model(modify_k_values=True)
    project_continuous_time_logistic_model(sinusoid=True, modify_initial_pop=True, sinusoid_k0=200, sinusoid_k1=100)
    project_continuous_time_logistic_model(sinusoid=True, modify_k_values=True)
    project_continuous_time_logistic_model(sinusoid=True, modify_initial_pop=True, modify_k_values=True)
    project_continuous_time_logistic_model(sinusoid=True, modify_tp_value=True, sinusoid_k0=200, sinusoid_k1=100)
    project_continuous_time_logistic_model(sinusoid=True, modify_tp_value=True, sinusoid_k0=10, sinusoid_k1=2)

