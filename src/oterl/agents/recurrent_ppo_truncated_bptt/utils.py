from agents.recurrent_ppo_truncated_bptt.environments.abides_gym import AbidesGym
from agents.recurrent_ppo_truncated_bptt.environments.cartpole_env import CartPole

def create_env(config:dict, render:bool=False):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        config {dict}: The configuration of the environment.

    Returns:
        {env}: Returns the selected environment instance.
        environment:
    type: "Abides"
    starting_cash: 10000000
    timestep_duration: "5S"
    order_fixed_size: 20
    execution_window: "00:30:00"
    parent_order_size: 10000
    debug_mode: True

    """
    if config["type"] == "CartPole":
        return CartPole(mask_velocity=False)
    if config["type"] == "CartPoleMasked":
        return CartPole(mask_velocity=True, realtime_mode = render)
    if config["type"] == "Abides":
        return AbidesGym(
            starting_cash=config["starting_cash"],
            timestep_duration=config["timestep_duration"],
            order_fixed_size=config["order_fixed_size"],
            execution_window=config["execution_window"],
            parent_order_size=config["parent_order_size"],
            debug_mode=config["debug_mode"],
        )
    
def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)