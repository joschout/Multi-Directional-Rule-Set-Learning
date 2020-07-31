from distributed import Client


def main_close_ssh_cluster(scheduler_host: str, scheduler_port: int):
    scheduler_address = f'{scheduler_host}:{scheduler_port}'
    client = Client(address=scheduler_address)
    client.shutdown()
    print("Shutting down client, its connected scheduler and workers...")
