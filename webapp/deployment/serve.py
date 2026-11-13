"""Portable entry point: one API process owns the durable queue."""
import os


def main():
    import uvicorn
    uvicorn.run('PALM.webapp.app:app', host=os.environ.get('WEBAPP_BIND_HOST', '0.0.0.0'),
                port=int(os.environ.get('PORT', '7860')), workers=1,
                proxy_headers=True, forwarded_allow_ips=os.environ.get('FORWARDED_ALLOW_IPS', '127.0.0.1'))


if __name__ == '__main__':
    main()
