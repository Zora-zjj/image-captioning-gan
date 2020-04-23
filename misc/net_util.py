def remove_module(state):
    new_state = {}
    for key, val in state.items():
        new_state[key[key.index(".") + 1:]] = val   #str.index(str, beg=0, end=len(string))  检测字符串中是否包含子字符串 str 
    return new_state
