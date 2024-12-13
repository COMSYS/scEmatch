from src.scmap_client_server import host


class Server(host.Host):

    def __init__(
            self, server_name, write_directory,
            client1_name, client1_write_directory,
            client2_name, client2_write_directory,
            benchmark_collector
    ):
        super().__init__(name=server_name, write_directory=write_directory, benchmark_collector=benchmark_collector)

        # Information of client1.
        self.client1_name = client1_name
        self.client1_write_directory = client1_write_directory

        # Information of client2.
        self.client2_name = client2_name
        self.client2_write_directory = client2_write_directory
