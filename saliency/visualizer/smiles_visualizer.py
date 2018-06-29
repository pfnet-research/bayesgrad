import numpy

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


from saliency.visualizer.base_visualizer import BaseVisualizer


def _convert_to_2d(axes, nrows, ncols):
    if nrows == 1 and ncols == 1:
        axes = numpy.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]
    else:
        pass
    assert axes.ndim == 2
    return axes


def is_visible(begin, end):
    if begin <= 0 or end <= 0:
        return 0
    elif begin >= 1 or end >= 1:
        return 1
    else:
        return (begin + end) * 0.5


# Default color function
def red(x):
    # return in RGB order
    # x=0 -> 1, 1, 1 (white)
    # x=1 -> 1, 0, 0 (red)
    return 1., 1. - x, 1. - x


def min_max_scaler(saliency):
    """Normalize saliency to value 0-1"""
    maxv = numpy.max(saliency)
    minv = numpy.min(saliency)
    if maxv == minv:
        saliency = numpy.zeros_like(saliency)
    else:
        saliency = (saliency - minv) / (maxv - minv)
    return saliency


class SmilesVisualizer(BaseVisualizer):

    def visualize(self, saliency, smiles, save_filepath=None,
                  visualize_ratio=1.0, color_fn=red, scaler=min_max_scaler, legend=''):
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()
        rdDepictor.Compute2DCoords(mol)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        n_atoms = mol.GetNumAtoms()
        # highlight = list(range(n_atoms))

        # --- type check ---
        assert saliency.ndim == 1
        # Cut saliency array for unnecessary tail part
        saliency = saliency[:num_atoms]
        # Normalize to [0, 1]
        saliency = scaler(saliency)
        # normed_saliency = copy.deepcopy(saliency)

        if visualize_ratio < 1.0:
            threshold_index = int(n_atoms * visualize_ratio)
            idx = numpy.argsort(saliency)
            idx = numpy.flip(idx, axis=0)
            # set threshold to top `visualize_ratio` saliency
            threshold = saliency[idx[threshold_index]]
            saliency = numpy.where(saliency < threshold, 0., saliency)
        else:
            threshold = numpy.min(saliency)

        highlight_atoms = list(map(lambda g: g.__int__(), numpy.where(saliency >= threshold)[0]))
        atom_colors = {i: color_fn(e) for i, e in enumerate(saliency)}
        bondlist = [bond.GetIdx() for bond in mol.GetBonds()]

        def color_bond(bond):
            begin = saliency[bond.GetBeginAtomIdx()]
            end = saliency[bond.GetEndAtomIdx()]
            return color_fn(is_visible(begin, end))
        bondcolorlist = {i: color_bond(bond) for i, bond in enumerate(mol.GetBonds())}
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 375)
        drawer.DrawMolecule(
            mol, highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors, highlightBonds=bondlist,
            highlightBondColors=bondcolorlist, legend=legend)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        if save_filepath:
            extention = save_filepath.split('.')[-1]
            if extention == 'svg':
                print('saving svg to {}'.format(save_filepath))
                with open(save_filepath, 'w') as f:
                    f.write(svg)
            elif extention == 'png':
                import cairosvg
                print('saving png to {}'.format(save_filepath))
                # cairosvg.svg2png(
                #     url=svg_save_filepath, write_to=save_filepath)
                # print('svg type', type(svg))
                cairosvg.svg2png(bytestring=svg, write_to=save_filepath)
            else:
                raise ValueError(
                    'Unsupported extention {} for save_filepath {}'
                    .format(extention, save_filepath))
        else:
            from IPython.core.display import SVG
            return SVG(svg.replace('svg:', ''))
