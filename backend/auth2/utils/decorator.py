def public_view(view_func):
    view_func.is_public = True
    return view_func