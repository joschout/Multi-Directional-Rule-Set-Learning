from logging import Logger
from typing import List, Tuple, Dict

import distributed
from dask.delayed import Delayed
from distributed import Future, Client

from experiments.utils.experiment_logging import create_logger, close_logger


def compute_delayed_functions(
        list_of_computations: List[Tuple[Delayed, Dict]],
        client: Client,
        nb_of_retries_if_erred: int,
        error_logger_name: str,
        error_logger_file_name: str
) -> None:
    print("start compute")
    print(list_of_computations)

    list_of_delayed_function_calls = [computation[0] for computation in list_of_computations]

    list_of_futures: List[Future] = client.compute(list_of_delayed_function_calls, retries=nb_of_retries_if_erred)
    distributed.wait(list_of_futures)
    print("end compute")

    error_logger: Logger = create_logger(logger_name=error_logger_name, log_file_name=error_logger_file_name)
    future: Future
    for future, (delayed, func_args) in zip(list_of_futures, list_of_computations):
        if future.status == 'error':
            exception = future.exception()
            error_logger.error(f"{exception.__class__}: {exception}\n"
                               f"\tfor arguments {func_args}"
                               )
    close_logger(error_logger)
