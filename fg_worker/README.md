# fg_worker (stage 7)

Separate executable hosting NVIDIA Streamline DLSS-G with a hidden swapchain and readback.
Isolated in its own process because Streamline hooks the swapchain and can take the host down;
the core talks to it over IPC. Nothing here until stage 7.
