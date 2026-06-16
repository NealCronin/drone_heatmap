import os
import socket
import subprocess
import time
import signal


class LlamaServer:

    def __init__(
        self,
        model_path: str,
        mmproj_path: str = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = -1,  # 99 ensures full offload to Metal/Apple Silicon
        ctx_size: int = 2048,
        binary_path: str = "llama-server",  # Adjust if not globally in PATH
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.binary_path = binary_path
        self.process = None

    def is_port_in_use(self) -> bool:
        """Checks if the target port is already occupied."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0

    def start(self, timeout: int = 30) -> bool:
        """Launches the llama.cpp server in a separate background process."""
        if self.is_port_in_use():
            print(
                f"[!] Port {self.port} is already in use. Assuming server is alive."
            )
            return True

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LLM model file not found at {self.model_path}"
            )

        # Base server command
        cmd = [
            self.binary_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-m",
            self.model_path,
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.ctx_size),
        ]

        # Append multimodal projector if using a VLM (like Qwen-VL)
        if self.mmproj_path:
            if not os.path.exists(self.mmproj_path):
                raise FileNotFoundError(
                    f"Multimodal projector file not found at {self.mmproj_path}"
                )
            cmd.extend(["--mmproj", self.mmproj_path])

        print(f"[*] Starting llama.cpp server: {' '.join(cmd)}")

        # Launch the process isolated from the main Python process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # Creates a process group for clean termination
        )

        # Wait for the HTTP server to spin up and accept connections
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_port_in_use():
                print(
                    f"[+] Server successfully started on http://{self.host}:{self.port}"
                )
                return True
            time.sleep(0.5)

        # If it timed out, check if it failed immediately to dump errors
        return_code = self.process.poll()
        if return_code is not None:
            _, stderr = self.process.communicate()
            print(f"[-] Server failed to start (Exit code {return_code}).")
            print(f"[-] Error log:\n{stderr}")
        else:
            print("[-] Server startup timed out.")

        return False

    def stop(self):
        """Cleanly terminates the server process group."""
        if self.process and self.process.poll() is None:
            print(f"[*] Shutting down llama.cpp server on port {self.port}...")
            try:
                # Send SIGTERM to the entire process group to ensure clean exit
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
                print("[+] Server stopped cleanly.")
            except subprocess.TimeoutExpired:
                print("[-] Server hung on SIGTERM. Forcing SIGKILL...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        else:
            print("[!] No active server instance tracked by this manager.")


# --- Example Usage Implementation ---
if __name__ == "__main__":
    # Define your local file paths
    MODEL = "/Users/neal/Documents/llama/llama.cpp/models/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
    PROJECTOR = "/Users/neal/Documents/llama/llama.cpp/models/Qwen3VL8B-mmproj-F32.gguf"

    server = LlamaServer(
        model_path=MODEL, mmproj_path=PROJECTOR, port=8600
    )

    try:
        if server.start():
            print("[*] Server is running! Doing other pipeline tasks here...")
            time.sleep(5)  # Simulate running your SAM inference or other tasks
    finally:
        server.stop()