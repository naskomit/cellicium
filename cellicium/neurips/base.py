import collections as coll
import scanpy as sc

ProblemDatasetBase = coll.namedtuple(
    'ProblemDataset', [
        'train_mod1',
        'train_mod2',
        'train_sol',
        'test_mod1',
        'test_mod2',
        'test_sol',
        'modality1',
        'modality2'
    ]
)

class ProblemDataset(ProblemDatasetBase):
    __slots__ = ()
    def to_modality_dict(self, part = 'train'):
        if part == 'train':
            return {self.modality1: self.train_mod1, self.modality2: self.train_mod2}
        elif part == 'test':
            return {self.modality1: self.test_mod1, self.modality2: self.test_mod2}
        else:
            raise ValueError(f'part should be train or test, not {part}')

    @property
    def _combined_mod1(self):
        result = sc.concat([self.train_mod1, self.test_mod1], axis = 0)
        result.uns['modality'] = self.modality1
        return result

    @property
    def _combined_mod2(self):
        result = sc.concat([self.train_mod2, self.test_mod2], axis = 0)
        result.uns['modality'] = self.modality2
        return result

    def get_data(self, group, modality):
        if modality == self.modality1:
            if group == 'train':
                return self.train_mod1
            elif group == 'test':
                return self.test_mod1
            else:
                raise ValueError(f'Group must be train or test, not {group}')
        elif modality == self.modality2:
            if group == 'train':
                return self.train_mod2
            elif group == 'test':
                return self.test_mod2
            else:
                raise ValueError(f'Group must be train or test, not {group}')
        else:
            raise ValueError(f'Modality must be {self.modality1} or {self.modality2}, not {modality}')