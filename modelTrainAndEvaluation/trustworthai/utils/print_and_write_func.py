def print_and_write(filename, message, clear_file=False, newline=2):
    """
    appends a message to a given file, and adds the given number of newlines
    after the message. If the file doesn't exist it is created.
    if clear_file is True, the given file is first cleared before the message
    is written
    will convert tensor arrays to nicely written lists
    """
    try:
        if ('torch' in message.type()) or ('Torch' in message.type):
            if len(message.shape) == 0:
                message = str(message.numpy())
            else:
                message = str(list(message.numpy()))
    except:
        pass # message isnt a torch object, just use pretty-printer as normal
        
    message = str(message)
    mode = 'w' if clear_file else 'a'
    with open(filename, mode) as f:
        f.write(message)
        for _ in range(newline):
            f.write("\n")
    print(message)
        