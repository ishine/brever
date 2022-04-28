import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['lines.markersize'] = 1


def _pol_to_car(r, phi, center):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    x += center[0]
    y += center[1]
    return x, y


def _car_to_pol(x, y, center):
    b, a = x-center[0], y-center[1]
    r = np.sqrt(a**2 + b**2)
    phi = np.arctan2(b, a)
    return r, phi


class RandomGenerator:

    def _check_size(self, size):
        flatten_output = False
        if size is None:
            size = 1
            flatten_output = True
        return size, flatten_output

    def _flatten(self, x, y, bool_):
        if bool_:
            x, y = x.flatten(), y.flatten()
        return x, y

    def square(
        self,
        center=(0.0, 0.0),
        width=1.0,
        height=1.0,
        size=None,
    ):
        size, flatten_output = self._check_size(size)
        x = np.random.uniform(0.0, width, size)
        y = np.random.uniform(0.0, height, size)
        x += center[0] - width/2
        y += center[1] - height/2
        return self._flatten(x, y, flatten_output)

    def circle(
        self,
        center=(0.0, 0.0),
        radius=1.0,
        size=None,
    ):
        size, flatten_output = self._check_size(size)
        r = np.sqrt(np.random.uniform(0, radius, size))
        phi = np.random.uniform(0, 2*np.pi, size)
        x, y = _pol_to_car(r, phi, center)
        return self._flatten(x, y, flatten_output)

    def anti_circle(
        self,
        center=(0.0, 0.0),
        radius=0.5,
        height=1.0,
        width=1.0,
        size=None,
    ):
        size, flatten_output = self._check_size(size)
        x = np.random.uniform(0.0, width, size)
        y = np.random.uniform(0.0, height, size)
        x += center[0] - width/2
        y += center[1] - height/2
        r, phi = _car_to_pol(x, y, center)
        mask = r > radius
        x, y = x[mask], y[mask]
        return self._flatten(x, y, flatten_output)

    def moon(
        self,
        outer_center=(0.0, 0.0),
        outer_radius=1.0,
        inner_center=(0.33, 0.0),
        inner_radius=0.83,
        size=None,
    ):
        size, flatten_output = self._check_size(size)
        x, y = np.empty(size), np.empty(size)
        for i in range(size):
            x[i], y[i] = self.circle(outer_center, outer_radius)
            r, phi = _car_to_pol(x[i], y[i], inner_center)
            while r < inner_radius:
                x[i], y[i] = self.circle(outer_center, outer_radius)
                r, phi = _car_to_pol(x[i], y[i], inner_center)
        return self._flatten(x, y, flatten_output)


class Drawer:

    def circle(
        self,
        center=(0.0, 0.0),
        radius=1.0,
        size=100,
    ):
        phi = np.linspace(0, 2*np.pi, size)
        return _pol_to_car(radius, phi, center)

    def moon(
        self,
        outer_center=(0.0, 0.0),
        outer_radius=1.0,
        inner_center=(0.33, 0.0),
        inner_radius=0.83,
        size=100,
    ):
        # draw outer circle
        x_out, y_out = self.circle(outer_center, outer_radius, size)
        r, phi = _car_to_pol(x_out, y_out, inner_center)
        mask = r > inner_radius
        x_out, y_out = x_out[mask], y_out[mask]
        # draw inner circle
        x_in, y_in = self.circle(inner_center, inner_radius, size)
        r, phi = _car_to_pol(x_in, y_in, outer_center)
        mask = r < outer_radius
        x_in, y_in = x_in[mask], y_in[mask]
        # concatenate
        x = np.concatenate((x_out, x_in[::-1]))
        y = np.concatenate((y_out, y_in[::-1]))
        return x, y


def main():
    gen = RandomGenerator()
    draw = Drawer()

    fig, axes = plt.subplots(1, 3, figsize=(8, 5))

    r = 1.0
    h = 5
    w = 3
    density = 250.0
    n_circle = round(3.1416*r**2*density)
    n_square = round(h*w*density)
    in_r = 0.67
    in_c = (0.4, 0.0)
    moon_area = np.pi*(1 - in_r**2)  # roughly
    n_moon = round(moon_area*density)
    n_anti_circle = n_square-n_circle

    accuracies = []

    ax = axes[0]
    x, y = gen.anti_circle(radius=r, height=h, width=w, size=n_anti_circle)
    x2, y2 = gen.moon(inner_center=in_c, inner_radius=in_r, size=n_moon)
    x, y = np.concatenate((x, x2)), np.concatenate((y, y2))
    l1, = ax.plot(x, y, '.')
    x, y = gen.moon(inner_center=in_c, inner_radius=in_r, size=n_moon)
    l2, = ax.plot(x, y, '.')
    accuracy = (n_anti_circle+n_moon)/(n_anti_circle+n_moon+n_moon)
    accuracies.append((accuracy, accuracy))

    ax = axes[1]
    x, y = gen.anti_circle(radius=r, height=h, width=w, size=n_anti_circle)
    ax.plot(x, y, '.')
    x, y = gen.moon(inner_center=in_c, inner_radius=in_r, size=n_moon)
    ax.plot(x, y, '.')
    accuracies.append((1.0, 1.0))

    ax = axes[2]
    x, y = gen.square(height=h, width=w, size=n_square)
    ax.plot(x, y, '.')
    x, y = gen.circle(radius=r, size=n_circle)
    ax.plot(x, y, '.')
    acc1 = (n_anti_circle + n_circle)/(n_square+n_circle)
    acc2 = (n_anti_circle + n_moon)/(n_square+n_circle)
    accuracies.append((acc1, acc2))

    for ax, accuracy in zip(axes.flatten(), accuracies):
        x, y = draw.circle(radius=r+0.02)
        l1, = ax.plot(x, y)
        x, y = draw.moon(inner_center=in_c, inner_radius=in_r)
        l2, = ax.plot(x, y)

        ax.axis('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        ax.legend([l1, l2], [
            f'Accuracy: {round(accuracy[0]*100)}%',
            f'Accuracy: {round(accuracy[1]*100)}%',
        ], loc="lower right")

    axes[0].set_title(
        '(a) Both models are trained on a dataset\nwhere the realizations '
        'do not fully\nrepresent the true distribution',
        fontsize=8,
    )
    axes[1].set_title(
        '(b) The red model can show a higher\nscore even though it did not '
        'learn the\nright distribution, if the test dataset is\neasier '
        'compared to (a)',
        fontsize=8,
    )
    axes[2].set_title(
        '(c) When presented with the real\ndistribution, '
        'the red model collapses',
        fontsize=8,
    )

    handles = ax.get_lines()
    labels = [
        "Class 0",
        "Class 1",
        "Robust decision boundary",
        "Naive decision boundary",
    ]
    fig.legend(handles, labels, loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    plt.show()


if __name__ == '__main__':
    main()
