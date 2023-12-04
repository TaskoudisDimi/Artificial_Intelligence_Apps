from App import app as application

if __name__ == '__main__':
    import wfastcgi
    wfastcgi.enable(debug=True)