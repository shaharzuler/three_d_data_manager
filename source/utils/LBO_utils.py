
import torch
import numpy as np
from scipy.sparse.linalg import eigsh, lobpcg
from scipy import sparse
from argparse import Namespace
from scipy.sparse import csr_matrix


def _reshape_and_repeat(a_array):
    """
    For a given 1-D array A, run the MATLAB code below.

        M = reshape(M,1,1,nfaces);
        M = repmat(M,3,3);

    Please note that a0 is a 3-D matrix, but the 3rd index in NumPy
    is the 1st index in MATLAB.  Fortunately, nfaces is the size of A.

    """
    return np.array([np.ones((3, 3)) * x for x in a_array])

def _compute_mats_ab(points, faces):
    """
    Compute matrices for the Laplace-Beltrami operator.

    The matrices correspond to A and B from Reuter's 2009 article.

    Note ::
        All points must be on faces. Otherwise, a singular matrix error
        is generated when inverting D.

    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex

    faces : list of lists of 3 integers
        each list contains indices to vertices that form a triangle on a mesh

    Returns
    -------
    A : csr_matrix
    B : csr_matrix
    """

    points = np.array(points)
    faces = np.array(faces)
    nfaces = faces.shape[0]

    # Linear local matrices on unit triangle:
    tB = (np.ones((3, 3)) + np.eye(3)) / 24.0

    tA00 = np.array([[0.5, -0.5, 0.0],
                     [-0.5, 0.5, 0.0],
                     [0.0, 0.0, 0.0]])

    tA11 = np.array([[0.5, 0.0, -0.5],
                     [0.0, 0.0, 0.0],
                     [-0.5, 0.0, 0.5]])

    tA0110 = np.array([[1.0, -0.5, -0.5],
                       [-0.5, 0.0, 0.5],
                       [-0.5, 0.5, 0.0]])

    # Replicate into third dimension for each triangle
    # (for tB, 1st index is the 3rd index in MATLAB):
    tB = np.array([np.tile(tB, (1, 1)) for i in range(nfaces)])
    tA00 = np.array([np.tile(tA00, (1, 1)) for i in range(nfaces)])
    tA11 = np.array([np.tile(tA11, (1, 1)) for i in range(nfaces)])
    tA0110 = np.array([np.tile(tA0110, (1, 1)) for i in range(nfaces)])

    # Compute vertex coordinates and a difference vector for each triangle:
    v1 = points[faces[:, 0], :]
    v2 = points[faces[:, 1], :]
    v3 = points[faces[:, 2], :]
    v2mv1 = v2 - v1
    v3mv1 = v3 - v1

    # Compute length^2 of v3mv1 for each triangle:
    a0 = np.sum(v3mv1 * v3mv1, axis=1)
    a0 = _reshape_and_repeat(a0)

    # Compute length^2 of v2mv1 for each triangle:
    a1 = np.sum(v2mv1 * v2mv1, axis=1)
    a1 = _reshape_and_repeat(a1)

    # Compute dot product (v2mv1*v3mv1) for each triangle:
    a0110 = np.sum(v2mv1 * v3mv1, axis=1)
    a0110 = _reshape_and_repeat(a0110)

    # Compute cross product and 2*vol for each triangle:
    cr = np.cross(v2mv1, v3mv1)
    vol = np.sqrt(np.sum(cr*cr, axis=1))
    # zero vol will cause division by zero below, so set to small value:
    vol_mean = 0.001*np.mean(vol)
    vol = [vol_mean if x == 0 else x for x in vol]
    vol = _reshape_and_repeat(vol)

    # Construct all local A and B matrices (guess: for each triangle):
    localB = vol * tB
    localA = (1.0/vol) * (a0*tA00 + a1*tA11 - a0110*tA0110)

    # Construct row and col indices.
    # (Note: J in numpy is I in MATLAB after flattening,
    #  because numpy is row-major while MATLAB is column-major.)
    J = np.array([np.tile(x, (3, 1)) for x in faces])
    I = np.array([np.transpose(np.tile(x, (3, 1))) for x in faces])

    # Flatten arrays and swap I and J:
    J_new = I.flatten()
    I_new = J.flatten()
    localA = localA.flatten()
    localB = localB.flatten()

    # Construct sparse matrix:
    A = sparse.csr_matrix((localA, (I_new, J_new)))
    B = sparse.csr_matrix((localB, (I_new, J_new)))

    return A, B

def _area_normalize(areas, spectrum):
    """
    Normalize a spectrum using areas as suggested in Reuter et al. (2006)

    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex of the structure
    faces : list of lists of 3 integers
        3 indices to vertices that form a triangle on the mesh
    spectrum : list of floats
        LB spectrum of a given shape defined by _points_ and _faces_

    Returns
    -------
    new_spectrum : list of floats
        LB spectrum normalized by area

    """
    total_area = sum(areas)

    new_spectrum = [x*total_area for x in spectrum]

    return new_spectrum

def _index_normalize(spectrum):
    """
    Normalize a spectrum by division of index to account for linear increase of
    Eigenvalue magnitude (Weyl's law in 2D) as suggested in Reuter et al. (2006)
    and used in BrainPrint (Wachinger et al. 2015)

    Parameters
    ----------
    spectrum : list of floats
        LB spectrum of a given shape

    Returns
    -------
    new_spectrum : list of floats
        Linearly re-weighted LB spectrum

    """

    # define index list of floats
    idx = [float(i) for i in range(1, len(spectrum) + 1)]
    # if first entry is zero, shift index
    if (abs(spectrum[0] < 1e-09)):
        idx = [i-1 for i in idx]
        idx[0] = 1.0
    # divide each element by its index
    new_spectrum = [x/i for x, i in zip(spectrum, idx)]

    return new_spectrum

def fem_laplacian(points, faces, spectrum_size=10, normalization="areaindex",
                  areas=None, verbose=False):
    """
    Compute linear finite-element method Laplace-Beltrami spectrum
    after Martin Reuter's MATLAB code.

    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex of the structure
    faces : list of lists of 3 integers
        3 indices to vertices that form a triangle on the mesh
    spectrum_size : integer
        number of eigenvalues to be computed (the length of the spectrum)
    normalization : string
        the method used to normalize eigenvalues
        if None, no normalization is used
        if "area", use area of the 2D structure as in Reuter et al. 2006
        if "index", divide eigenvalue by index to account for linear trend
        if "areaindex", do both (default)
    areas : list of triangles areas for area normalization
    verbose : bool
        print statements?

    Returns
    -------
    results : dict() contains:
        self_values : list
        first spectrum_size eigenvalues for Laplace-Beltrami spectrum
        self_functions : matrix of first spectrum self funcs
        mass_matrix : mass matrix (B from rauters)

    """
    result = dict()
    # ----------------------------------------------------------------
    # Compute A and B matrices (from Reuter et al., 2009):
    # ----------------------------------------------------------------
    A, B = _compute_mats_ab(points, faces)
    if A.shape[0] <= spectrum_size:
        if verbose:
            print("The 3D shape has too few vertices ({0} <= {1}). Skip.".
                  format(A.shape[0], spectrum_size))
        return None

    # ----------------------------------------------------------------
    # Use the eigsh eigensolver:
    # ----------------------------------------------------------------
    try:

        # eigs is for nonsymmetric matrices while
        # eigsh is for real-symmetric or complex-Hermitian matrices:
        # Martin Reuter: "small sigma shift helps prevent numerical
        #   instabilities with zero eigenvalue"
        eigenvalues, eigenvectors = eigsh(A, k=spectrum_size, M=B,
                                          sigma=-0.01)
        spectrum = eigenvalues.tolist()

    # ----------------------------------------------------------------
    # Use the lobpcg eigensolver:
    # ----------------------------------------------------------------
    except RuntimeError:

        if verbose:
            print("eigsh() failed. Now try lobpcg.")
            print("Warning: lobpcg can produce different results from "
                  "Reuter (2006) shapeDNA-tria software.")
        # Initial eigenvector values:
        init_eigenvecs = np.random.random((A.shape[0], spectrum_size))

        # maxiter = 40 forces lobpcg to use 20 iterations.
        # Strangely, largest=false finds largest eigenvalues
        # and largest=True gives the smallest eigenvalues:
        eigenvalues, eigenvectors = lobpcg(
            A, init_eigenvecs, B=B, largest=True, maxiter=40)
        # Extract the real parts:
        spectrum = [value.real for value in eigenvalues]

        # For some reason, the eigenvalues from lobpcg are not sorted:
        spectrum.sort()

    # ----------------------------------------------------------------
    # Normalize by area:
    # ----------------------------------------------------------------
    if normalization == "area":
        spectrum = _area_normalize(areas, spectrum)
        if verbose:
            print("Compute area-normalized linear FEM Laplace-Beltrami "
                  "spectrum")
    elif normalization == "index":
        spectrum = _index_normalize(spectrum)
        if verbose:
            print("Compute index-normalized linear FEM Laplace-Beltrami"
                  " spectrum")
    elif normalization == "areaindex":
        spectrum = _index_normalize(spectrum)
        spectrum = _area_normalize(areas, spectrum)
        if verbose:
            print("Compute area and index-normalized linear FEM "
                  "Laplace-Beltrami spectrum")
    else:
        if verbose:
            print("Compute linear FEM Laplace-Beltrami spectrum")
    result['self_values'] = spectrum
    result['self_functions'] = eigenvectors
    result['mass_matrix'] = B

    return result




class LBOcalc(object):
    def __init__(self, k=30, use_torch=False, is_point_cloud=False):
        self.k = k
        self.use_torch = use_torch
        self.is_pc = is_point_cloud

    def reproject(self, projected, faces, all_vects, prefix=""):
        all_vects_pseudo_inv = np.linalg.pinv(all_vects[0,:,:])
        reprojected = np.dot(projected, all_vects_pseudo_inv).transpose()
        dir_path = f"mesh_files/projection_exp_{self.k}" ###
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # visualize_grid_of_lbo(reprojected,faces,all_vects[0,:,:],dirpath=dir_path, prefix=prefix+"lbo_after_proj",max_lbos=4)

    def get_LBOs(self, verts, faces):   
        all_vects = []
        all_vals = []
        all_a = []

        shape_0 = Namespace(**{"pos": verts, "face": faces})
        try:
            shape_0 = self.__call__(shape_0) 
            all_vects.append(shape_0.evecs)
            all_vals.append(shape_0.evals)
            all_a.append(np.diag(shape_0.A))
        except:
            all_vects.append(np.zeros((verts.shape[0], self.k)))
            all_a.append(np.zeros(verts.shape[0]))
            all_vals.append(np.zeros(self.k))
            print('Error in frame')

        all_vects = np.array(all_vects) 
        all_a = np.array(all_a) 
        return all_vects[0], all_vals[0], all_a[0]

    def LBO_mesh(self, sample):
        S = dict()
        pos = sample.pos if hasattr(sample, 'pos') else sample.vert
        face = sample.face
        if (torch.is_tensor(pos)):
            pos = sample.pos.detach().cpu().numpy()
            face = sample.face.detach().cpu().numpy()
        if (face.shape[1] != 3):
            face = face.transpose()

        S["X"] = pos.transpose()[0]
        S["Y"] = pos.transpose()[1]
        S["Z"] = pos.transpose()[2]
        S["VERTS"] = pos
        S["TRIV"] = face
        S["nv"] = pos.shape[0]

        evals, evecs, evecs_trans, L, A = self.S_info(S, self.k, use_torch=self.use_torch)
        sample.evals = evals
        sample.evecs = evecs
        sample.evecs_trans = evecs_trans
        sample.Laplacian = L.todense()
        sample.A = A.todense()
        return sample

    def __call__(self, sample):
        return self.LBO_mesh(sample) if not self.is_pc else self.LBO_pc(sample)

    @staticmethod
    def cotLaplacian(S):
        T1 = S['TRIV'][:, 0]
        T2 = S['TRIV'][:, 1]
        T3 = S['TRIV'][:, 2]

        V1 = S['VERTS'][T1, :]
        V2 = S['VERTS'][T2, :]
        V3 = S['VERTS'][T3, :]

        L1 = np.linalg.norm(V2 - V3, axis=1)
        L2 = np.linalg.norm(V1 - V3, axis=1)
        L3 = np.linalg.norm(V1 - V2, axis=1)
        L = np.column_stack((L1, L2, L3))  # Edges of each triangle

        Cos1 = (L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3)
        Cos2 = (L1 ** 2 + L3 ** 2 - L2 ** 2) / (2 * L1 * L3)
        Cos3 = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2)
        Cos = np.column_stack((Cos1, Cos2, Cos3))  # Cosines of opposite edges for each triangle
        Ang = np.arccos(Cos)  # Angles

        I = np.concatenate((T1, T2, T3))
        J = np.concatenate((T2, T3, T1))
        w = 0.5 * LBOcalc.cotangent(np.concatenate((Ang[:, 2], Ang[:, 0], Ang[:, 1]))).astype(float)
        In = np.concatenate((I, J, I, J))
        Jn = np.concatenate((J, I, I, J))
        wn = np.concatenate((-w, -w, w, w))
        W = csr_matrix((wn, (In, Jn)), [S['nv'], S['nv']])  # Sparse Cotangent Weight Matrix

        cA = LBOcalc.cotangent(Ang) / 2  # Half cotangent of all angles
        At = 1 / 4 * (L[:, [1, 2, 0]] ** 2 * cA[:, [1, 2, 0]] + L[:, [2, 0, 1]] ** 2 * cA[:, [2, 0, 1]]).astype(
            float)  # Voronoi Area

        N = np.cross(V1 - V2, V1 - V3)
        Ar = np.linalg.norm(N, axis=1)  # Barycentric Area

        # Use Ar is ever cot is negative instead of At
        locs = cA[:, 0] < 0
        At[locs, 0] = Ar[locs] / 4;
        At[locs, 1] = Ar[locs] / 8;
        At[locs, 2] = Ar[locs] / 8;

        locs = cA[:, 1] < 0
        At[locs, 0] = Ar[locs] / 8;
        At[locs, 1] = Ar[locs] / 4;
        At[locs, 2] = Ar[locs] / 8;

        locs = cA[:, 2] < 0
        At[locs, 0] = Ar[locs] / 8;
        At[locs, 1] = Ar[locs] / 8;
        At[locs, 2] = Ar[locs] / 4;

        Jn = np.zeros(I.shape[0])
        An = np.concatenate((At[:, 0], At[:, 1], At[:, 2]))
        Area = csr_matrix((An, (I, Jn)), [S['nv'], 1])  # Sparse Vector of Area Weights

        In = np.arange(S['nv'])
        A = csr_matrix((np.squeeze(np.array(Area.todense())), (In, In)),
                    [S['nv'], S['nv']])  # Sparse Matrix of Area Weights
        return W, A

    @staticmethod
    def trim_tensor_values(tensor, vmin=-0.99,vmax=0.99):
        tensor[np.where(tensor>vmax)]=vmax 
        tensor[np.where(tensor<vmin)]=vmin
        return tensor

    @staticmethod
    def laplacian_torch(V, F,to_dense=False,to_numpy=False):
        if(isinstance(V, (np.ndarray))):
            V,F = torch.from_numpy(V.copy()),torch.from_numpy(F.copy()).long()
        nv = V.shape[0]

        T1 = F[:, 0]
        T2 = F[:, 1]
        T3 = F[:, 2]

        V1 = V[T1, :]
        V2 = V[T2, :]
        V3 = V[T3, :]

        L1 = torch.norm(V2-V3, dim=1)
        L2 = torch.norm(V1-V3, dim=1)
        L3 = torch.norm(V1-V2, dim=1)
        L = torch.stack([L1, L2, L3], dim=1) # Edges of each triangle

        Cos1 = (L2**2+L3**2-L1**2)/(2*L2*L3)
        Cos2 = (L1**2+L3**2-L2**2)/(2*L1*L3)
        Cos3 = (L1**2+L2**2-L3**2)/(2*L1*L2)
        Cos = torch.stack([Cos1, Cos2, Cos3], dim=1)  # Cosines of opposite edges for each triangle
        Cos = LBOcalc.trim_tensor_values(Cos, vmin=-0.99,vmax=0.99)
        Ang = torch.acos(Cos)  # Angles

        I = torch.cat([T1, T2, T3])
        J = torch.cat([T2, T3, T1])
        w = 0.5 * LBOcalc.cotangent_torch(torch.cat([Ang[:, 2], Ang[:, 0], Ang[:, 1]]))
        In = torch.cat([I, J, I, J])
        Jn = torch.cat([J, I, I, J])
        wn = torch.cat([-w, -w, w, w])
        ##########3
        # wn += 1e-6
        ##############
        # Sparse Cotangent Weight Matrix
        A = torch.sparse.FloatTensor(torch.stack([In, Jn]), wn, [nv, nv])

        cA = LBOcalc.cotangent_torch(Ang)/2  # Half cotangent of all angles
        At = 1/4 * (L[:, [1, 2, 0]]** 2 * cA[:, [1, 2, 0]] +
                    L[:, [2, 0, 1]]** 2 * cA[:, [2, 0, 1]]) # Voronoi Area

        N = torch.cross(V1-V2, V1-V3)
        Ar = torch.norm(N, dim=1)  # Barycentric Area

        #Use Ar is ever cot is negative instead of At
        locs = cA[:, 0] < 0
        At[locs, 0] = Ar[locs] / 4
        At[locs, 1] = Ar[locs] / 8
        At[locs, 2] = Ar[locs] / 8

        locs = cA[:, 1] < 0
        At[locs, 0] = Ar[locs] / 8
        At[locs, 1] = Ar[locs] / 4
        At[locs, 2] = Ar[locs] / 8

        locs = cA[:, 2] < 0
        At[locs, 0] = Ar[locs] / 8
        At[locs, 1] = Ar[locs] / 8
        At[locs, 2] = Ar[locs] / 4

        Jn = torch.zeros(I.shape[0], dtype=torch.long, device=V.device)
        An = torch.cat([At[:, 0], At[:, 1], At[:, 2]])

        # Sparse Vector of Area Weights
        Area = torch.sparse.FloatTensor(torch.stack([I, Jn]), An, [nv, 1])
        n = V.shape[0]
        Area = torch.diag_embed(Area.to_dense().squeeze())

        if to_dense:
            A, Area = A.to_dense(), Area
        if to_numpy:
            A, Area = csr_matrix(A.to_dense().numpy()),csr_matrix(Area.numpy())
        return A, Area

    @staticmethod
    def eigs_WA(W, A, numEig):
        eigvals, eigvecs = eigsh(W, numEig, A, 1e-6)
        return eigvals, eigvecs

    @staticmethod
    def S_info(S, numEig,use_torch=False):
        W, A = LBOcalc.cotLaplacian(S) if not use_torch else LBOcalc.laplacian_torch(S['VERTS'],S['TRIV'],to_numpy=True)
        eigvals, eigvecs = LBOcalc.eigs_WA(W, A, numEig)
        eigvecs_trans = eigvecs.T * A
        return eigvals, eigvecs, eigvecs_trans, W,A

    @staticmethod
    def cotangent(p):
        return np.cos(p)/np.sin(p)

    @staticmethod
    def cotangent_torch(p):
        return torch.cos(p)/torch.sin(p)

