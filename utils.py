def get_channels_names(stream_info):
    description = stream_info.desc()
    ch = description.child("channels").first_child()
    ch_names = [ch.child_value("label")]
    n_chan = stream_info.channel_count()
    for _ in range(n_chan - 1):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value("label"))
    return ch_names
