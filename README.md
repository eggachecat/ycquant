Welcome to ycquant!
===================


Info
-----------
## Important

Please modify gentic.py in gplearn as follows:

changing the code block start at line 472 from:

```python
    if isinstance(self, RegressorMixin):
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
