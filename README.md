Welcome to ycquant!
===================


Info
-----------
## Use virtualenv on windows
make dir
> mkdir env
create your own virtualenv
> virtualenv env --no-site-packages
active virtualenv
> env\Scripts\activate.bat
install depended packages(some packages need to be installed manually on windows such as numpy+mkl)
> pip install -r requirements.txt



## Important

I have modified gentic.py in gplearn as follows:

changing the code block start at line 472 from:

```python
    if isinstance(self, RegressorMixin):
        # Find the best individual in the final generation
        self._program = self._programs[-1][np.argmin(fitness)]
```

to:

```python
    if isinstance(self, RegressorMixin):
        # Find the best individual in the final generation
        if self._metric.greater_is_better:
            self._program = self._programs[-1][np.argmax(fitness)]
        else:
            self._program = self._programs[-1][np.argmin(fitness)]

```

Or you can simply install
> pip install git+git://github.com/eggachecat/YCgplearn.git
