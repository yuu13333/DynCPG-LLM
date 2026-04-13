FIND_METHOD = """def findMethod(node: AstNode): Option[Method] = {
        var current: AstNode = node  // 不能用 Any
        while (current != null) {
          current match {
            case m: Method => return Some(m)  // 找到 Method
            case n: AstNode =>
              val parents = n.astParent  // inAst 返回 Iterator[AstNode]
              if (parents.nonEmpty) current = parents // 取唯一父节点
              else current = null
          }
        }
        None
      }"""

PATH_STRING = """paths.zipWithIndex.map { case (p, i) =>
        val chainInfo = p.elements.map { e =>
          findMethod(e) match {
            case Some(m) =>
              val file = m.filename
              val nodeLine = e.lineNumber
              val fulllineOpt = for {
                  nodeLine <- e.lineNumber
                  methodLine <- m.lineNumber
                  idx = nodeLine - methodLine
                  codeLines = m.code.split("\\n") if idx >= 0 && idx < codeLines.length} yield codeLines(idx)
              val methodName = m.name
                s"${e.label}: ${e.code} @ line ${e.lineNumber.getOrElse("?")}: '${fulllineOpt.getOrElse("").trim()}', method: '$methodName'"
            case None =>
              s"${e.label}: ${e.code} @ line ${e.lineNumber.getOrElse("?")}"
          }
        }.distinct.zipWithIndex.map { case (info, stepIdx) => s"[Step $stepIdx] $info"}.mkString("\\n")

        chainInfo
      }.distinct.zipWithIndex.map { case (chainInfo, newIdx) => s"Path $newIdx:\\n$chainInfo"}.l"""
