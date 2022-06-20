## Some Tips for the User in Resberry Pi

1. Make all ip address the same as your server address. For example, if you want your PC become the server for the FL, 
then make all the ip host to your PC's IP address. You should change your IP address in both client and server manager
in the FedML file in both sides.

2. Because we use the Mqtt as the communication between the device and server. We highly suggest you to install a Emqs on your PC
as the server of the Mqtt.

3. When you use the Mqtt, the port number should be 1883. When you use the HTTP in the app.py, the port should be 5000.

4. NEVER change the time.sleep(), it may trigger very serious bugs.

5. It now only supports the old version of FedML, the version that inside the FedIoT, pls do not upgrade it.

6. In this folder, you should put app.py in your PC and main_uci_rp.py in your Rasberry Pi. These two files could definitely work.

7. Pls believe that this lib could work, but it may take you a very long time to let it run.

## Instruction for the Mqtt

To run a Mqtt, you need to install a server first on your PC.

1. Download emqx to your PC. Run the following on your terminal.

```brew tap emqx/emqx```

```brew install emqx```

2. Activate your server, run the following on your terminal.

```emqx start```

3. You could see your server activity by http://127.0.0.1:18083. User name is admin, pwd is public.

4. When you stop your server, type the following on your terminal.

```emqx stop```
