from pypushdeer import PushDeer

pushdeer = PushDeer(pushkey="PDU28532TNPEt0t1KxjJSxc6L9OrF9YuMwtC8sx0B")
pushdeer.send_text("hello world", desp="optional description")
pushdeer.send_markdown("# hello world", desp="**optional** description in markdown")
pushdeer.send_image("https://github.com/easychen/pushdeer/raw/main/doc/image/clipcode.png")
pushdeer.send_image(
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQYV2NgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII=")