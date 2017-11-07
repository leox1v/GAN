import sys
import numpy as np
from PIL import Image
from utils import *
import mode_counting
from reportlab.pdfgen import canvas

dataset = "toy"
exp_no = 2
eval_no = [1, 4, 10]
eval_mode_collapse = False

def main(dataset=dataset, exp_no=exp_no):

    try:
        FLAGS = get_flags(dataset, exp_no)
    except:
        raise AttributeError("Flags have not been saved. So the parameters are lost.")

    optimizer = FLAGS.opt_methods

    exp_path = "results/{}/exp_{}/".format(dataset, exp_no)
    construct_img(exp_path, optimizer)

    if eval_mode_collapse:
        mode_counting.main(dataset, exp_no)

    #helper = Helper(FLAGS)
    #create_pdf(exp_path, helper)


def construct_img(exp_path, optimizer):
    horiz_imgs = []
    for opt in optimizer.split(" "):
        img_paths = [exp_path + "out/{0:s}_{1:0=3d}.png".format(opt, i) for i in eval_no]
        images = map(Image.open, img_paths)
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        images = map(Image.open, img_paths)
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        #new_im.save(exp_path + 'test_{}.jpg'.format(opt))

        horiz_imgs.append(new_im)
    width = np.max([im.size[0] for im in horiz_imgs])
    height = np.sum([im.size[1] for im in horiz_imgs])

    new_im = Image.new('RGB', (width, height))
    y_offset = 0
    for im in horiz_imgs:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    new_im.save(exp_path + 'Convergence.jpg')


import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as pdf_Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_pdf(exp_path, helper):
    doc = SimpleDocTemplate(exp_path + "Results.pdf", pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72,
                            bottomMargin=18)
    Story = []
    if os.path.isfile(exp_path + "Gradients_annotated.png"):
        gradients = exp_path + "Gradients_annotated.png"
    else:
        gradients = exp_path + "Gradients.png"
    convergence = exp_path + "Convergence.jpg"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    text = '<font size=22 style="align:center">Results</font>'
    Story.append(Paragraph(text, styles["Justify"]))
    Story.append(Spacer(1, 12))
    text = '<font size=12>{}</font>'.format(helper.get_info_string().replace("\n", "<br>"))
    Story.append(Paragraph(text, styles["Justify"]))

    im = pdf_Image(gradients, width=400, height=480)
    Story.append(im)
    im = pdf_Image(convergence, width=500, height=500)
    Story.append(im)





    # magName = "Pythonista"
    # issueNum = 12
    # subPrice = "99.00"
    # limitedDate = "03/05/2010"
    # freeGift = "tin foil hat"
    #
    # formatted_time = time.ctime()
    # full_name = "Mike Driscoll"
    # address_parts = ["411 State St.", "Marshalltown, IA 50158"]

    # styles = getSampleStyleSheet()
    # styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    # ptext = '<font size=12>%s</font>' % formatted_time
    #
    # Story.append(Paragraph(ptext, styles["Normal"]))
    # Story.append(Spacer(1, 12))
    #
    # # Create return address
    # ptext = '<font size=12>%s</font>' % full_name
    # Story.append(Paragraph(ptext, styles["Normal"]))
    # for part in address_parts:
    #     ptext = '<font size=12>%s</font>' % part.strip()
    #     Story.append(Paragraph(ptext, styles["Normal"]))
    #
    # Story.append(Spacer(1, 12))
    # ptext = '<font size=12>Dear %s:</font>' % full_name.split()[0].strip()
    # Story.append(Paragraph(ptext, styles["Normal"]))
    # Story.append(Spacer(1, 12))
    #
    # ptext = '<font size=12>We would like to welcome you to our subscriber base for %s Magazine! \
    #         You will receive %s issues at the excellent introductory price of $%s. Please respond by\
    #         %s to start receiving your subscription and get the following free gift: %s.</font>' % (
    # magName, issueNum, subPrice, limitedDate, freeGift)
    # Story.append(Paragraph(ptext, styles["Justify"]))
    # Story.append(Spacer(1, 12))
    #
    # ptext = '<font size=12>Thank you very much and we look forward to serving you.</font>'
    # Story.append(Paragraph(ptext, styles["Justify"]))
    # Story.append(Spacer(1, 12))
    # ptext = '<font size=12>Sincerely,</font>'
    # Story.append(Paragraph(ptext, styles["Normal"]))
    # Story.append(Spacer(1, 48))
    # ptext = '<font size=12>Ima Sucker</font>'
    # Story.append(Paragraph(ptext, styles["Normal"]))
    # Story.append(Spacer(1, 12))
    doc.build(Story)


if __name__ == '__main__':
    main()