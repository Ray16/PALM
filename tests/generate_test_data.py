"""Generate test datasets in various formats for integration testing.

Creates:
  tests/data/molecules.sdf      — multi-molecule SDF file (20 molecules)
  tests/data/sdf_dir/           — directory of individual SDF files
  tests/data/cif_dir/           — directory of material CIF files
  tests/data/sequences.fasta    — FASTA file with protein sequences
  tests/data/molecules.smi      — SMILES file

Usage:
    python -m tests.generate_test_data
"""

import os

TESTS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTS_DIR, "data")


def generate_sdf():
    """Generate a multi-molecule SDF file and an SDF directory."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles_list = [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("naproxen", "COc1ccc2cc(CC(C)C(=O)O)ccc2c1"),
        ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
        ("metformin", "CN(C)C(=N)NC(=N)N"),
        ("omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
        ("atorvastatin", "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(c2ccc(F)cc2)c(c1c1ccccc1)C(=O)Nc1ccccc1"),
        ("losartan", "CCCCc1nc(Cl)c(n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)CO"),
        ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
        ("penicillin_v", "CC1(C)[C@@H](N2C(=O)[C@@H](NC(=O)COc3ccccc3)[C@H]2S1)C(=O)O"),
        ("ciprofloxacin", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
        ("amoxicillin", "CC1(C)[C@@H](N2C(=O)[C@@H](NC(=O)[C@@H](N)c3ccc(O)cc3)[C@H]2S1)C(=O)O"),
        ("morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=C[C@@H]1[C@@H]3C=C5)O"),  # changed to simplified
        ("codeine", "COc1ccc2C[C@H]3N(C)CC[C@@]45c2c1OC4C(=CC3)CC5O"),  # simplified
        ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
        ("benzocaine", "CCOC(=O)c1ccc(N)cc1"),
        ("dopamine", "NCCc1ccc(O)c(O)c1"),
        ("serotonin", "NCCc1c[nH]c2ccc(O)cc12"),
    ]

    # Multi-molecule SDF file
    sdf_path = os.path.join(DATA_DIR, "molecules.sdf")
    writer = Chem.SDWriter(sdf_path)
    for name, smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
        mol.SetProp("_Name", name)
        mol.SetProp("category", "drug")
        writer.write(mol)
    writer.close()
    print(f"  Created {sdf_path} ({len(smiles_list)} molecules)")

    # SDF directory (one molecule per file)
    sdf_dir = os.path.join(DATA_DIR, "sdf_dir")
    os.makedirs(sdf_dir, exist_ok=True)
    for name, smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol.SetProp("_Name", name)
        fpath = os.path.join(sdf_dir, f"{name}.sdf")
        w = Chem.SDWriter(fpath)
        w.write(mol)
        w.close()
    print(f"  Created {sdf_dir}/ ({len(smiles_list)} files)")


def generate_cif_dir():
    """Generate a directory of material CIF files using ASE."""
    from ase import Atoms
    from ase.io import write as ase_write

    cif_dir = os.path.join(DATA_DIR, "cif_dir")
    os.makedirs(cif_dir, exist_ok=True)

    # Simple cubic crystals with varying compositions
    materials = [
        ("NaCl", ["Na", "Cl"], [(0, 0, 0), (0.5, 0.5, 0.5)], 5.64),
        ("MgO", ["Mg", "O"], [(0, 0, 0), (0.5, 0.5, 0.5)], 4.21),
        ("CaO", ["Ca", "O"], [(0, 0, 0), (0.5, 0.5, 0.5)], 4.81),
        ("LiF", ["Li", "F"], [(0, 0, 0), (0.5, 0.5, 0.5)], 4.03),
        ("KCl", ["K", "Cl"], [(0, 0, 0), (0.5, 0.5, 0.5)], 6.29),
        ("BaO", ["Ba", "O"], [(0, 0, 0), (0.5, 0.5, 0.5)], 5.52),
        ("SrO", ["Sr", "O"], [(0, 0, 0), (0.5, 0.5, 0.5)], 5.16),
        ("CsF", ["Cs", "F"], [(0, 0, 0), (0.5, 0.5, 0.5)], 6.01),
        ("RbCl", ["Rb", "Cl"], [(0, 0, 0), (0.5, 0.5, 0.5)], 6.58),
        ("KBr", ["K", "Br"], [(0, 0, 0), (0.5, 0.5, 0.5)], 6.60),
        ("NaF", ["Na", "F"], [(0, 0, 0), (0.5, 0.5, 0.5)], 4.63),
        ("LiCl", ["Li", "Cl"], [(0, 0, 0), (0.5, 0.5, 0.5)], 5.13),
        ("CaF2", ["Ca", "F", "F"], [(0, 0, 0), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)], 5.46),
        ("BaF2", ["Ba", "F", "F"], [(0, 0, 0), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)], 6.20),
        ("SrF2", ["Sr", "F", "F"], [(0, 0, 0), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)], 5.80),
        ("TiO2", ["Ti", "O", "O"], [(0, 0, 0), (0.3, 0.3, 0.0), (0.7, 0.7, 0.0)], 4.59),
        ("ZnO", ["Zn", "O"], [(0, 0, 0), (0.33, 0.33, 0.5)], 3.25),
        ("Fe2O3", ["Fe", "Fe", "O", "O", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.3, 0, 0.25), (0, 0.3, 0.25), (0.7, 0.7, 0.25)], 5.04),
        ("Al2O3", ["Al", "Al", "O", "O", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.3, 0, 0.25), (0, 0.3, 0.25), (0.7, 0.7, 0.25)], 4.76),
        ("Cu2O", ["Cu", "Cu", "O"], [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0, 0, 0)], 4.27),
    ]

    for name, symbols, scaled_pos, a in materials:
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=scaled_pos,
            cell=[a, a, a],
            pbc=True,
        )
        fpath = os.path.join(cif_dir, f"{name}.cif")
        ase_write(fpath, atoms, format="cif")

    print(f"  Created {cif_dir}/ ({len(materials)} CIF files)")


def generate_fasta():
    """Generate a FASTA file with short protein sequences."""
    sequences = [
        ("kinase_1", "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPD"),
        ("kinase_2", "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHQRQETKISREQFMELLKRIDHENVIKEACKVFHQNTEREYLDKIRDLGEDHFKYLVYRESMEGKLRDRIQILTDRILQPESILAKEIRDPERIKIYELSINKLD"),
        ("receptor_1", "MQSGTHWRVLGLCLLSVGVWGQDGNEEMGGITQTPYKVSISGTTVILTCPQYPGSEILWQHNDKNIGGDEDDKNIGSDEDHLSLKEFSELEQSGYYVCYPRGSKPEDANFYLYLRARVCENCMEMDVMSVATIVIVDICITGGLLLLVYYYWSKNRKAKAKPTNRDRNNKEEYSAPATMPQE"),
        ("receptor_2", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYRETHPIFPAIIGAKLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"),
        ("enzyme_1", "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEG"),
        ("enzyme_2", "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"),
        ("transporter_1", "MAGKPKRPSSAEGAMEISKSSQETPESLDTFNAAFDKTSAFGSNNPFPNAAAANSTSSGPPRPAHTASQNNLNNSNLNNNLSGKHDENISETSTTHSSDSGINASEANAHRGMTPKAANRGSSMDLFDYDGTLAGLLEQRDGSLFPAKIIMFEAPRGECFHFALGREDAWDYAKLHRIGEEKFEPR"),
        ("transporter_2", "MEVFCGAALTLLLVLGLQGTLAQPGPELDCPFHDQPHLENKNATFPCEEEEERITWDKHMKQDGHYIIAPDVEKNKNVTCTTQYNARLEGKDGPPLMHCSAGWDKGQLDKCVNLSGKWNDEHCSQRLPYICKQTPNLAPKCLKEFQNLSQELQDTQKELDKYVGGLELPDTQKYVGR"),
        ("channel_1", "MTSRRWFKSLPPTFLMTALASWRSYSGGWEKPQSDQELLPRGAQALAHPAPSEETPGEAGGGGPQALAQVFRSQFPQGRLDTRPKAFEISQFPAVTPRPLAGSGKFLGTFIPNYLGGDGGDISMIPFAAVLIFSTIIAIMALGYYWLASHRGKQKDRADDKPSNKQ"),
        ("channel_2", "MDELFQELDTTHYTLADRTSSGTYSPMKPMSMSYQNRSYRSLRMQMAAALGFSHKRPQKRGFIKEEHGEFYEAYESPFRLAELEVHNQYNVSAHSYARIKNERTIMRGASFKMDKAKGEMDYYPFSDLASSLVAATPEELGGFWELYLKDHILAQFELSMEADKLNLREIQKIWEKMKPEAEF"),
        ("protease_1", "MHSFWPRALPSLLLFLLCSRLPGSEAHCGATCDENECLFDKHTCSMACCLLGPQDASQDEAVCNCFINEQNTVCKDECRKANQCSYCKESETQPSYPCHTCPSGYRYANADSCECTPF"),
        ("protease_2", "MKLLALTLYLCSLTCTQAQEEEDAPKVLMRGEITHLGQPVKFYKHYALRQGTQAAQLLKPGDTLPRQAAAGSLSSPLLQFTSAIGMKLHNLIHSQERNYNIHPILHHNAVSAVKLVNEKAVKSYRKAYPAKTKEQYQQLYNEASRLLQESAAEVAEKVLANQGELDNCSGHFHQLEKGRYSGKATLNYLNKTLAPRKYGKDTHFQEIRVRVQKESLVHSF"),
        ("signaling_1", "MESKKGHQLFVKNVTDLKSGIAPRANFLKPMPEEEYDFKTLKAMEGNHLIQWSESLEKEDWKSSTHSDDTKTHTSEETIKPRSQMEGLSDSEVFLSKLKKGTKACYHLFETKACNTSGCFEECLKKRLQKKSDSKKKSEQRQMAEKEKEFVDELDASTAQLEAQILELKELKLNREEHKK"),
        ("signaling_2", "MFMKTLGAVILLITGWTFDAAQKQFQFNFEKGEEWRGTVSDHITLRDSYEAHHQVSPTGNYDEKFKQYPEGIAVHPVALRLKDMPALNSGHMTSGMQKGQKVLYEYEKRYPRSQHSSDTTKPMPEETQAAKIFEKLNQPFDDMQAEKAKEEFGKLDAAPPATTDHKEAAKLFNQDVD"),
        ("binding_1", "MRGMLPLLALLLLVGTQSAAADTICIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDSHNGKLCRLKGIAPLQLGKCNIAGWILGNPECESLFSKKSWSYIAETPNSENGTCYPGYFADYEELREQLSSVSSFERFEIFPKESSWPNHNTNGVTAACSHEGKSSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPPTTADQQSLYQNADAYV"),
        ("binding_2", "MLTDFKNPATFATDSMDQLAHFLSPMADEEGCVQPEEKDSCEGYSRDEPRGVYQITSNPEQNMSHCALDYATGGPNQQDRFCLERWLYNLLPGDKPNFTIKHRFECKDSAHMSECIRDVNANDNPDIEVFSKKGQKLGNSGIYGVSSRYKGVDTSGRMVSGIRDMQITATDKKYKNFKNTRITYKVTTQEVSWASVYYGGAEANRNLVPSNVHINKITKIQNPLLGN"),
        ("structural_1", "MCDEDETTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEK"),
        ("structural_2", "MSSTKKKVSACDISAGQKLVSADEGSYQAMQFGKNTFAGLALKDAAAVPKFFGQHIAKQGGVHPNIQAVFAKYLDDECIDIVTHTDEHALLETHQMFDAKVIASIDEMNADLEQAASEATKHTNRITSYLPAGQNPVQKECIVRIAHNLGRSHLAHANTDAWFALKFRSQIKEATETKSEEVKAVLRNTKVFVHAHIEE"),
        ("chaperone_1", "MAKKTAIGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVALNPQNTVFDAKRLIGRKFGDPVVQSDMKHWPFQVINDGDKPKVQVSYKGETKAFYPEEISSMVLTKMKEIAEAYLGYPVTNAVITVPAYFNDSQRQATKDAGVIAGLNVLRIINEPTAAAIAYGLDK"),
        ("chaperone_2", "MPEEVHHGEEEVETFAFQAEIAQLMSLIINTFYSNKEIFLRELISNASDALDKIRYESLTDPSKLDSGKELHINLIPNKQDRTLTIVDTGIGMTKADLINNLGTIAKSGTKAFMEALQAGADISMIGQFGVGFYSAYLVAEKVTVITKHNDDEQYAWESSAGGSFTVRTDTGEPMGRGTKVILHLKEDQTEYLEERRIKEIVKKHSQFIGYPITLFVEK"),
    ]

    fasta_path = os.path.join(DATA_DIR, "sequences.fasta")
    with open(fasta_path, "w") as fh:
        for seq_id, seq in sequences:
            fh.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")
    print(f"  Created {fasta_path} ({len(sequences)} sequences)")


def generate_smiles():
    """Generate a SMILES file."""
    molecules = [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("naproxen", "COc1ccc2cc(CC(C)C(=O)O)ccc2c1"),
        ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
        ("metformin", "CN(C)C(=N)NC(=N)N"),
        ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
        ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
        ("benzocaine", "CCOC(=O)c1ccc(N)cc1"),
        ("dopamine", "NCCc1ccc(O)c(O)c1"),
        ("serotonin", "NCCc1c[nH]c2ccc(O)cc12"),
        ("histamine", "NCCc1cnc[nH]1"),
        ("adrenaline", "CNC[C@H](O)c1ccc(O)c(O)c1"),
        ("melatonin", "COc1ccc2[nH]cc(CCNC(C)=O)c2c1"),
        ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
        ("sucrose", "OC[C@H]1OC(O[C@@]2(CO)OC(CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
        ("cholesterol", "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C"),
        ("vanillin", "COc1cc(C=O)ccc1O"),
        ("capsaicin", "COc1cc(CNC(=O)CCCC\\C=C\\C(C)C)ccc1O"),
    ]

    smi_path = os.path.join(DATA_DIR, "molecules.smi")
    with open(smi_path, "w") as fh:
        for name, smi in molecules:
            fh.write(f"{smi} {name}\n")
    print(f"  Created {smi_path} ({len(molecules)} molecules)")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating test datasets...")
    generate_sdf()
    generate_cif_dir()
    generate_fasta()
    generate_smiles()
    print("Done!")
