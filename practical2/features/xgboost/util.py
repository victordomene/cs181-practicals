# sets of syscalls
all_syscalls = ['processes', 'process', 'thread', 'all_section', 'load_image', 'load_dll', 'open_file', 'get_windows_directory', 'check_for_debugger', 'get_system_directory', 'open_key', 'query_value', 'create_mutex', 'set_windows_hook', 'create_window', 'find_window', 'enum_window', 'show_window', 'get_file_attributes', 'create_directory', 'create_thread', 'sleep', 'destroy_window', 'find_file', 'com_create_instance', 'vm_protect', 'enum_keys', 'enum_values', 'com_get_class_object', 'create_process', 'kill_process', 'create_file', 'set_file_time', 'set_file_attributes', 'open_process', 'delete_file', 'create_key', 'delete_value', 'read_value', 'read_section', 'set_value', 'remove_directory', 'get_computer_name', 'impersonate_user', 'open_scmanager', 'get_host_by_name', 'create_socket', 'create_open_file', 'bind_socket', 'connect_socket', 'send_socket', 'dump_line', 'recv_socket', 'trimmed_bytes', 'open_url', 'write_value', 'open_mutex', 'open_service', 'get_system_time', 'connect', 'enum_processes', 'copy_file', 'get_username', 'delete_key', 'revert_to_self', 'move_file', 'enum_share', 'vm_allocate', 'vm_write', 'create_thread_remote', 'message', 'listen_socket', 'enum_modules', 'download_file', 'create_service', 'change_service_config', 'start_service', 'set_thread_context', 'vm_read', 'create_interface', 'enum_types', 'enum_subtypes', 'enum_items', 'load_driver', 'control_service', 'create_namedpipe', 'add_netjob', 'download_file_to_cache', 'unload_driver', 'com_createole_object', 'create_mailslot', 'create_process_as_user', 'delete_service', 'logon_as_user', 'get_host_by_addr', 'create_process_nt', 'enum_services', 'get_userinfo', 'read_section_names', 'set_system_time', 'vm_mapviewofsection', 'delete_share', 'enum_handles', 'accept_socket', 'enum_user', 'exit_windows']

likely_syscalls = ['dump_line', 'sleep', 'load_dll', 'vm_protect', 'open_key', 'query_value', 'create_file', 'get_file_attributes',
    'com_create_instance', 'open_file', 'find_file', 'enum_processes', 'create_mutex', 'set_windows_hook', 'get_host_by_name',
    'process', 'thread']

# these are the fifteen malware classes we're looking for
malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

# a function for writing predictions in the required format
def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    index of the predicted class with the malware_classes list above for 
    the executable corresponding to ids[i].
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%d\n" % (history_id, predictions[i]))

# a function for writing predictions in the required format
def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    index of the predicted class with the malware_classes list above for 
    the executable corresponding to ids[i].
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%d\n" % (history_id, predictions[i]))