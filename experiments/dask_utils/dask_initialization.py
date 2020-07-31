from typing import List

from dask.distributed import Client, SSHCluster

scheduler_host_name = 'scheduler_host_name'
worker_hosts = [
    'worker1',
    'worker2'
]


def initialize_client_for_ssh_cluster(
        scheduler_host: str,
        worker_hosts: List[str]
) -> Client:
    ssh_hosts = [scheduler_host, *worker_hosts]
    try:
        cluster = SSHCluster(
            hosts=ssh_hosts,
            connect_options={"known_hosts": None},
            worker_options={"nthreads": 1},
            # scheduler_options={"port": 0, "dashboard_address": ":8787"}
        )
        client = Client(cluster)
    except (KeyError, OSError):
        scheduler_address = f'{scheduler_host}:8786'
        client = Client(address=scheduler_address)

    return client


def reconnect_client_to_ssh_cluster(scheduler_host: str) -> Client:
    scheduler_address = f'{scheduler_host}:8786'
    client = Client(address=scheduler_address)

    return client
